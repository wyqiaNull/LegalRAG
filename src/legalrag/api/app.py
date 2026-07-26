"""LegalRAG FastAPI 应用。"""

from __future__ import annotations

import hashlib
import time
import uuid
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from redis import Redis
from redis.exceptions import RedisError

from .. import __version__
from ..application import (
    build_vector_store,
    new_usage_collector,
    run_contract_review,
    run_query_details,
)
from ..config.settings import Settings, load_settings
from ..core.errors import (
    ConfigError,
    ContractReviewError,
    GenerationError,
    IngestError,
    LegalRAGError,
    RetrievalError,
    StorageError,
)
from ..core.models import Confidentiality, DocType
from .auth import Principal, principal_dependency, require_scopes
from .repository import ServiceRepository
from .schemas import (
    AuditRecord,
    ContractReviewResponse,
    DeleteDocumentResponse,
    DocumentPage,
    FeedbackRequest,
    FeedbackResponse,
    IngestAccepted,
    QueryRequest,
    QueryResponse,
    TaskStatusResponse,
)
from .tasks import TaskDispatcher

_SUPPORTED_INGEST = {".pdf", ".docx", ".txt"}
_SUPPORTED_CONTRACT = {".pdf", ".docx", ".txt"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def get_repository(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ServiceRepository:
    return ServiceRepository(settings.secrets.postgres_dsn)


def get_dispatcher(
    settings: Annotated[Settings, Depends(get_settings)],
) -> TaskDispatcher:
    return TaskDispatcher(settings)


get_principal = principal_dependency(get_settings)
query_principal = require_scopes(get_principal, "query")
ingest_principal = require_scopes(get_principal, "ingest")
contract_principal = require_scopes(get_principal, "contract:review")
feedback_principal = require_scopes(get_principal, "feedback")
documents_read_principal = require_scopes(get_principal, "documents:read")
documents_delete_principal = require_scopes(get_principal, "documents:delete")


def _http_error(error: LegalRAGError) -> HTTPException:
    if isinstance(error, (ConfigError, IngestError)):
        return HTTPException(status_code=400, detail="请求配置或输入无效")
    if isinstance(error, ContractReviewError):
        return HTTPException(status_code=422, detail="合同审查输入或结果无效")
    if isinstance(error, StorageError):
        return HTTPException(status_code=503, detail="存储服务暂不可用")
    if isinstance(error, (RetrievalError, GenerationError)):
        return HTTPException(status_code=502, detail="模型服务调用失败")
    return HTTPException(status_code=500, detail="服务内部错误")


def _error_code(error: Exception) -> str:
    mapping = {
        ConfigError: "config_error",
        IngestError: "ingest_error",
        StorageError: "storage_error",
        RetrievalError: "retrieval_error",
        GenerationError: "generation_error",
        ContractReviewError: "contract_review_error",
    }
    return next(
        (code for error_type, code in mapping.items() if isinstance(error, error_type)),
        "internal_error",
    )


async def _save_upload(
    upload: UploadFile,
    settings: Settings,
    supported: set[str],
) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in supported:
        raise HTTPException(status_code=422, detail="不支持的文件类型")
    directory = Path(settings.config.service.upload_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{uuid.uuid4()}{suffix}"
    size = 0
    try:
        with target.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > settings.config.service.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="上传文件超过大小限制")
                output.write(chunk)
        if size == 0:
            raise HTTPException(status_code=422, detail="上传文件不能为空")
        return str(target)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="LegalRAG API",
        version=__version__,
        description="企业合同与法规问答服务",
    )

    @app.exception_handler(LegalRAGError)
    async def legalrag_error_handler(_, error: LegalRAGError):
        mapped = _http_error(error)
        return JSONResponse(status_code=mapped.status_code, content={"detail": mapped.detail})

    @app.get("/health/live", tags=["health"])
    async def liveness() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    async def readiness(
        settings: Annotated[Settings, Depends(get_settings)],
        repository: Annotated[ServiceRepository, Depends(get_repository)],
    ) -> dict[str, str]:
        checks = {"postgres": repository.ping(), "redis": False, "milvus": False}
        try:
            broker = settings.secrets.celery_broker_url or settings.secrets.redis_url
            checks["redis"] = bool(broker and Redis.from_url(broker).ping())
        except (RedisError, ValueError):
            pass
        try:
            vector_store = await run_in_threadpool(build_vector_store, settings)
            checks["milvus"] = bool(getattr(vector_store, "ping", lambda: False)())
        except LegalRAGError:
            pass
        if not all(checks.values()):
            raise HTTPException(status_code=503, detail={"status": "not_ready", **checks})
        return {"status": "ready"}

    @app.post("/ingest", response_model=IngestAccepted, status_code=202)
    async def ingest(
        file: Annotated[UploadFile, File()],
        doc_name: Annotated[str, Form(min_length=1, max_length=512)],
        version: Annotated[str, Form(min_length=1, max_length=128)],
        doc_type: Annotated[DocType, Form()] = DocType.REGULATION,
        department: Annotated[str, Form(max_length=256)] = "",
        allowed_roles: Annotated[list[str] | None, Form()] = None,
        confidentiality: Annotated[Confidentiality, Form()] = Confidentiality.PUBLIC,
        effective_date: Annotated[str | None, Form()] = None,
        global_document: Annotated[bool, Form()] = False,
        principal: Annotated[Principal, Depends(ingest_principal)] = None,
        settings: Annotated[Settings, Depends(get_settings)] = None,
        dispatcher: Annotated[TaskDispatcher, Depends(get_dispatcher)] = None,
    ) -> IngestAccepted:
        if global_document and "admin:global" not in principal.scopes:
            raise HTTPException(status_code=403, detail="缺少全局文档管理权限")
        normalized_date = None
        if effective_date:
            try:
                normalized_date = date.fromisoformat(effective_date).isoformat()
            except ValueError as exc:
                raise HTTPException(status_code=422, detail="生效日期格式必须为 YYYY-MM-DD") from exc
        path = await _save_upload(file, settings, _SUPPORTED_INGEST)
        tenant_id = settings.config.governance.shared_tenant_id if global_document else principal.tenant_id
        doc_meta = {
            "doc_name": doc_name,
            "doc_type": doc_type.value,
            "tenant_id": tenant_id,
            "department": department,
            "allowed_roles": ["*"] if global_document else (allowed_roles or [principal.role]),
            "confidentiality": (
                Confidentiality.PUBLIC.value if global_document else confidentiality.value
            ),
            "version": version,
            "effective_date": normalized_date,
        }
        task_id = dispatcher.enqueue(
            path,
            doc_meta,
            principal,
            settings.secrets.legalrag_config,
        )
        return IngestAccepted(task_id=task_id)

    @app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
    async def task_status(
        task_id: str,
        principal: Annotated[Principal, Depends(ingest_principal)],
        dispatcher: Annotated[TaskDispatcher, Depends(get_dispatcher)],
    ) -> TaskStatusResponse:
        try:
            return dispatcher.status(task_id, principal)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail="异步任务不存在或已过期") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail="无权查看该异步任务") from exc

    @app.post("/query", response_model=QueryResponse)
    async def query_endpoint(
        request: QueryRequest,
        principal: Annotated[Principal, Depends(query_principal)],
        settings: Annotated[Settings, Depends(get_settings)],
        repository: Annotated[ServiceRepository, Depends(get_repository)],
    ) -> QueryResponse:
        request_id = str(uuid.uuid4())
        query_hash = hashlib.sha256(request.query.encode("utf-8")).hexdigest()
        started = time.perf_counter()
        usage = new_usage_collector(settings)
        try:
            result = await run_in_threadpool(
                run_query_details,
                request.query,
                settings,
                request.session_id,
                principal.identity(),
                request.version,
                usage,
            )
            elapsed = (time.perf_counter() - started) * 1000
            summary = usage.summary()
            chunk_ids = [candidate.chunk.chunk_id for candidate in result.candidates]
            repository.save_audit(
                AuditRecord(
                    request_id=request_id,
                    user_id=principal.user_id,
                    role=principal.role,
                    tenant_id=principal.tenant_id,
                    query_sha256=query_hash,
                    route=result.answer.route.value,
                    refused=result.answer.refused,
                    reason=result.answer.reason,
                    chunk_ids=chunk_ids,
                    latency_ms=elapsed,
                    status="succeeded",
                    prompt_tokens=summary.prompt_tokens,
                    completion_tokens=summary.completion_tokens,
                    total_tokens=summary.total_tokens,
                    estimated_cost_usd=summary.estimated_cost_usd,
                    usage_complete=summary.complete,
                )
            )
            return QueryResponse(
                request_id=request_id,
                answer=result.answer.text,
                citations=result.answer.citations,
                refused=result.answer.refused,
                reason=result.answer.reason,
                route=result.answer.route,
                retrieved_chunk_ids=chunk_ids,
            )
        except LegalRAGError as error:
            elapsed = (time.perf_counter() - started) * 1000
            summary = usage.summary()
            repository.save_audit(
                AuditRecord(
                    request_id=request_id,
                    user_id=principal.user_id,
                    role=principal.role,
                    tenant_id=principal.tenant_id,
                    query_sha256=query_hash,
                    latency_ms=elapsed,
                    status="failed",
                    error_code=_error_code(error),
                    prompt_tokens=summary.prompt_tokens,
                    completion_tokens=summary.completion_tokens,
                    total_tokens=summary.total_tokens,
                    estimated_cost_usd=summary.estimated_cost_usd,
                    usage_complete=summary.complete,
                )
            )
            raise _http_error(error) from error

    @app.post("/contract/review", response_model=ContractReviewResponse)
    async def contract_review_endpoint(
        file: Annotated[UploadFile, File()],
        principal: Annotated[Principal, Depends(contract_principal)],
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> ContractReviewResponse:
        path = await _save_upload(file, settings, _SUPPORTED_CONTRACT)
        try:
            report = await run_in_threadpool(
                run_contract_review,
                path,
                settings,
                principal.identity(),
            )
            return ContractReviewResponse(**report.model_dump(mode="python"))
        finally:
            Path(path).unlink(missing_ok=True)

    @app.post("/feedback", response_model=FeedbackResponse, status_code=201)
    async def feedback_endpoint(
        request: FeedbackRequest,
        principal: Annotated[Principal, Depends(feedback_principal)],
        repository: Annotated[ServiceRepository, Depends(get_repository)],
    ) -> FeedbackResponse:
        try:
            return FeedbackResponse(
                feedback_id=repository.create_feedback(principal, request)
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail="查询记录不存在") from exc

    @app.get("/admin/documents", response_model=DocumentPage)
    async def list_documents(
        principal: Annotated[Principal, Depends(documents_read_principal)],
        repository: Annotated[ServiceRepository, Depends(get_repository)],
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    ) -> DocumentPage:
        items = repository.list_documents(
            principal.tenant_id,
            include_global="admin:global" in principal.scopes,
            page=page,
            page_size=page_size,
        )
        return DocumentPage(items=items, page=page, page_size=page_size)

    @app.delete(
        "/admin/documents/{doc_id}",
        response_model=DeleteDocumentResponse,
    )
    async def delete_document(
        doc_id: str,
        principal: Annotated[Principal, Depends(documents_delete_principal)],
        settings: Annotated[Settings, Depends(get_settings)],
        repository: Annotated[ServiceRepository, Depends(get_repository)],
    ) -> DeleteDocumentResponse:
        chunk_ids = repository.document_chunk_ids(
            doc_id,
            principal.tenant_id,
            allow_global="admin:global" in principal.scopes,
        )
        if not chunk_ids:
            raise HTTPException(status_code=404, detail="文档不存在")
        vector_store = await run_in_threadpool(build_vector_store, settings)
        delete_chunks = getattr(vector_store, "delete_chunks", None)
        if not callable(delete_chunks):
            raise HTTPException(status_code=503, detail="当前向量存储不支持文档删除")
        await run_in_threadpool(delete_chunks, chunk_ids)
        repository.mark_document_deleted(doc_id, chunk_ids)
        return DeleteDocumentResponse(
            doc_id=doc_id,
            deleted_chunk_count=len(chunk_ids),
        )

    return app


app = create_app()
