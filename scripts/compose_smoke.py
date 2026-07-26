"""Docker Compose 内的确定性端到端验收。"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import httpx
import psycopg

from legalrag.api.auth import create_token
from legalrag.config.settings import load_settings
from legalrag.core.models import Confidentiality

_STATE_FILE = "compose-smoke-state.json"


def _token(settings, tenant_id: str) -> str:
    return create_token(
        settings,
        user_id="compose-smoke",
        role="legal_staff",
        tenant_id=tenant_id,
        allowed_confidentiality=list(Confidentiality),
        scopes={
            "query",
            "ingest",
            "feedback",
            "documents:read",
            "documents:delete",
        },
        token_id=str(uuid.uuid4()),
    )


def _wait_for_ingest(
    client: httpx.Client, task_id: str, headers: dict[str, str]
) -> dict[str, int]:
    deadline = time.monotonic() + 120
    while True:
        response = client.get(f"/tasks/{task_id}", headers=headers)
        response.raise_for_status()
        payload = response.json()
        if payload["status"] == "succeeded":
            result = payload.get("result") or {}
            if result.get("total_chunks", 0) < 1:
                raise RuntimeError("Compose 摄取统计缺少有效 chunk")
            return result
        if payload["status"] == "failed":
            raise RuntimeError(f"Compose 摄取任务失败：{payload['failure_reason']}")
        if time.monotonic() >= deadline:
            raise TimeoutError("等待 Compose 摄取任务超时")
        time.sleep(1)


def prepare() -> None:
    settings = load_settings()
    state_path = Path(settings.config.service.upload_dir) / _STATE_FILE
    if state_path.exists():
        raise RuntimeError("Compose smoke 状态文件已存在，请先完成或清理上次验收")
    tenant_id = f"company-smoke-{uuid.uuid4().hex}"
    marker = uuid.uuid4().hex
    doc_name = f"Compose验收法规-{marker}"
    query = f"工程验收编号{marker}的试用期最长多久？"
    headers = {"Authorization": f"Bearer {_token(settings, tenant_id)}"}

    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=30) as client:
        response = client.post(
            "/ingest",
            headers=headers,
            data={
                "doc_name": doc_name,
                "version": "v1",
                "doc_type": "regulation",
            },
            files={
                "file": (
                    "smoke.txt",
                    f"第一条 工程验收编号{marker}的试用期不得超过六个月。".encode(),
                )
            },
        )
        response.raise_for_status()
        _wait_for_ingest(client, response.json()["task_id"], headers)

        answer = client.post("/query", headers=headers, json={"query": query})
        answer.raise_for_status()
        query_payload = answer.json()
        if (
            query_payload["refused"]
            or not query_payload["retrieved_chunk_ids"]
            or not query_payload["citations"]
        ):
            raise RuntimeError("Compose 查询未返回已摄取法规及引用")

        feedback = client.post(
            "/feedback",
            headers=headers,
            json={"request_id": query_payload["request_id"], "value": "helpful"},
        )
        feedback.raise_for_status()

        documents = client.get("/admin/documents", headers=headers)
        documents.raise_for_status()
        document = next(
            item for item in documents.json()["items"] if item["doc_name"] == doc_name
        )

    state_path.write_text(
        json.dumps(
            {
                "tenant_id": tenant_id,
                "doc_id": document["doc_id"],
                "doc_name": doc_name,
                "query": query,
                "request_id": query_payload["request_id"],
                "chunk_ids": query_payload["retrieved_chunk_ids"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("Compose smoke prepare passed")


def verify() -> None:
    settings = load_settings()
    state_path = Path(settings.config.service.upload_dir) / _STATE_FILE
    if not state_path.exists():
        raise RuntimeError("找不到 Compose smoke prepare 状态")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    headers = {"Authorization": f"Bearer {_token(settings, state['tenant_id'])}"}
    other_tenant_id = f"other-{state['tenant_id']}"
    other_headers = {
        "Authorization": f"Bearer {_token(settings, other_tenant_id)}"
    }

    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=30) as client:
        documents = client.get("/admin/documents", headers=headers)
        documents.raise_for_status()
        if state["doc_id"] not in {item["doc_id"] for item in documents.json()["items"]}:
            raise RuntimeError("容器重启后 PostgreSQL 文档不存在")

        answer = client.post("/query", headers=headers, json={"query": state["query"]})
        answer.raise_for_status()
        if not set(state["chunk_ids"]) & set(answer.json()["retrieved_chunk_ids"]):
            raise RuntimeError("容器重启后 Milvus 未返回已摄取 chunk")

        denied = client.post(
            "/query",
            headers=other_headers,
            json={"query": state["query"]},
        )
        denied.raise_for_status()
        if set(state["chunk_ids"]) & set(denied.json()["retrieved_chunk_ids"]):
            raise RuntimeError("跨租户查询返回了越权候选")

        deleted = client.delete(
            f"/admin/documents/{state['doc_id']}", headers=headers
        )
        deleted.raise_for_status()

        after_delete = client.post(
            "/query", headers=headers, json={"query": state["query"]}
        )
        after_delete.raise_for_status()
        if set(state["chunk_ids"]) & set(after_delete.json()["retrieved_chunk_ids"]):
            raise RuntimeError("软删除后 Milvus 仍返回已删除 chunk")

    with psycopg.connect(settings.secrets.postgres_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT chunk_ids, status FROM query_audits WHERE request_id = %s",
                (state["request_id"],),
            )
            row = cursor.fetchone()
            if row is None or not row[0] or row[1] != "succeeded":
                raise RuntimeError("Compose 查询审计记录不完整")
            cursor.execute(
                "SELECT 1 FROM feedback WHERE request_id = %s AND user_id = %s",
                (state["request_id"], "compose-smoke"),
            )
            if cursor.fetchone() is None:
                raise RuntimeError("Compose 查询反馈记录不完整")
            cursor.execute(
                "SELECT deleted_at FROM documents WHERE doc_id = %s",
                (state["doc_id"],),
            )
            deleted_at = cursor.fetchone()
            if deleted_at is None or deleted_at[0] is None:
                raise RuntimeError("Compose 文档软删除记录不完整")

    state_path.unlink()
    print("Compose smoke verify passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "verify"))
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    else:
        verify()


if __name__ == "__main__":
    main()
