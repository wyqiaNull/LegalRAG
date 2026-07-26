"""Celery 摄取任务与 Redis 任务归属登记。"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from celery import Celery
from celery.result import AsyncResult
from redis import Redis
from redis.exceptions import RedisError

from ..application import run_ingest_details
from ..config.settings import Settings, load_settings
from ..core.errors import GenerationError, RetrievalError, StorageError
from .auth import Principal
from .schemas import TaskState, TaskStatusResponse

_TASK_OWNER_PREFIX = "legalrag:task-owner"
_TASK_OWNER_TTL_SECONDS = 7 * 24 * 60 * 60


def _celery_urls(settings: Settings) -> tuple[str, str]:
    secrets = settings.secrets
    broker = secrets.celery_broker_url or secrets.redis_url or "redis://127.0.0.1:6379/1"
    backend = (
        secrets.celery_result_backend
        or secrets.redis_url
        or "redis://127.0.0.1:6379/2"
    )
    if not broker.startswith("redis://") or not backend.startswith("redis://"):
        raise StorageError("LegalRAG Celery 任务归属登记要求使用 Redis broker/backend")
    return broker, backend


def create_celery_app(settings: Settings | None = None) -> Celery:
    settings = settings or load_settings()
    broker, backend = _celery_urls(settings)
    app = Celery("legalrag", broker=broker, backend=backend)
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        result_expires=_TASK_OWNER_TTL_SECONDS,
        timezone="UTC",
        enable_utc=True,
    )
    return app


celery_app = create_celery_app()


@celery_app.task(bind=True, name="legalrag.ingest_document", max_retries=3)
def ingest_document(
    self,
    path: str,
    doc_meta: dict[str, Any],
    config_path: str | None = None,
) -> dict[str, int]:
    retrying = False
    try:
        result = run_ingest_details(path, load_settings(config_path), **doc_meta)
        return {
            "total_chunks": result.stats.total_chunks,
            "embedded_chunks": result.stats.embedded_chunks,
            "reused_vectors": result.stats.reused_vectors,
            "superseded_chunks": result.stats.superseded_chunks,
        }
    except (StorageError, RetrievalError, GenerationError) as exc:
        if self.request.retries < self.max_retries:
            retrying = True
            raise self.retry(exc=exc, countdown=min(60, 2 ** self.request.retries))
        raise
    finally:
        if not retrying:
            Path(path).unlink(missing_ok=True)


class TaskDispatcher:
    def __init__(self, settings: Settings) -> None:
        broker, _ = _celery_urls(settings)
        self.redis = Redis.from_url(broker, decode_responses=True)

    @staticmethod
    def _owner_key(task_id: str) -> str:
        return f"{_TASK_OWNER_PREFIX}:{task_id}"

    def enqueue(
        self,
        path: str,
        doc_meta: dict[str, Any],
        principal: Principal,
        config_path: str | None,
    ) -> str:
        task_id = str(uuid.uuid4())
        owner = json.dumps(
            {"user_id": principal.user_id, "tenant_id": principal.tenant_id},
            separators=(",", ":"),
        )
        try:
            self.redis.setex(self._owner_key(task_id), _TASK_OWNER_TTL_SECONDS, owner)
            ingest_document.apply_async(
                args=[path, doc_meta, config_path],
                task_id=task_id,
            )
            return task_id
        except Exception as exc:
            try:
                self.redis.delete(self._owner_key(task_id))
            except RedisError:
                pass
            Path(path).unlink(missing_ok=True)
            raise StorageError("提交异步摄取任务失败") from exc

    def status(self, task_id: str, principal: Principal) -> TaskStatusResponse:
        try:
            raw_owner = self.redis.get(self._owner_key(task_id))
        except RedisError as exc:
            raise StorageError("读取异步任务归属失败") from exc
        if raw_owner is None:
            raise LookupError("异步任务不存在或已过期")
        owner = json.loads(raw_owner)
        if owner != {"user_id": principal.user_id, "tenant_id": principal.tenant_id}:
            raise PermissionError("无权查看该异步任务")

        result = AsyncResult(task_id, app=celery_app)
        state_map = {
            "PENDING": TaskState.PENDING,
            "RECEIVED": TaskState.RUNNING,
            "STARTED": TaskState.RUNNING,
            "RETRY": TaskState.RETRYING,
            "SUCCESS": TaskState.SUCCEEDED,
            "FAILURE": TaskState.FAILED,
            "REVOKED": TaskState.FAILED,
        }
        state = state_map.get(result.state, TaskState.FAILED)
        value = result.result if state is TaskState.SUCCEEDED else None
        return TaskStatusResponse(
            task_id=task_id,
            status=state,
            result=value if isinstance(value, dict) else None,
            error_code="ingest_failed" if state is TaskState.FAILED else None,
            failure_reason=(
                "摄取任务执行失败" if state is TaskState.FAILED else None
            ),
        )
