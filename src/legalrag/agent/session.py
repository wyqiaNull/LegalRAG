"""按身份隔离的 JSON 与 Redis 会话存储。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from redis import Redis
from redis.exceptions import RedisError

from ..core import registry
from ..core.errors import ConfigError, StorageError
from ..core.interfaces import ConversationStore
from ..core.models import Identity


def conversation_key(identity: Identity, session_id: str, key_prefix: str) -> str:
    """生成不会因分隔符冲突而跨身份串话的存储键。"""
    prefix = key_prefix.strip().strip(":")
    if not prefix:
        raise ConfigError("conversation.key_prefix 不能为空")
    parts = (identity.tenant_id, identity.user_id, session_id)
    encoded = [quote(part, safe="") for part in parts]
    return ":".join([prefix, *encoded])


class JsonConversationStore(ConversationStore):
    def __init__(
        self,
        path: str,
        history_turns: int = 4,
        key_prefix: str = "legalrag:conversation",
    ) -> None:
        self.path = Path(path)
        self.history_turns = history_turns
        self.key_prefix = key_prefix

    def _load(self) -> dict[str, list[str]]:
        if not self.path.exists():
            return {}
        raw: Any = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        return {
            str(session_id): [str(question) for question in questions]
            for session_id, questions in raw.items()
            if isinstance(questions, list)
        }

    def get(self, identity: Identity, session_id: str) -> list[str]:
        key = conversation_key(identity, session_id, self.key_prefix)
        return self._load().get(key, [])

    def append(
        self,
        identity: Identity,
        session_id: str,
        standalone_question: str,
    ) -> None:
        sessions = self._load()
        key = conversation_key(identity, session_id, self.key_prefix)
        history = sessions.setdefault(key, [])
        history.append(standalone_question)
        sessions[key] = history[-self.history_turns :]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.path)


class RedisConversationStore(ConversationStore):
    def __init__(
        self,
        url: str = "",
        history_turns: int = 4,
        ttl_seconds: int = 604800,
        key_prefix: str = "legalrag:conversation",
        client: Redis | None = None,
    ) -> None:
        if history_turns < 1:
            raise ConfigError("conversation.history_turns 必须至少为 1")
        if ttl_seconds < 1:
            raise ConfigError("conversation.ttl_seconds 必须至少为 1")
        if client is None and not url:
            raise ConfigError("Redis 会话存储需配置 REDIS_URL")
        self.history_turns = history_turns
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
        try:
            self.client = client or Redis.from_url(url, decode_responses=True)
            if not self.client.ping():
                raise StorageError("Redis 会话存储连接失败：PING 未返回成功")
        except (RedisError, ValueError) as error:
            raise StorageError(f"Redis 会话存储连接失败：{error}") from error

    def get(self, identity: Identity, session_id: str) -> list[str]:
        key = conversation_key(identity, session_id, self.key_prefix)
        try:
            return [str(item) for item in self.client.lrange(key, 0, -1)]
        except RedisError as error:
            raise StorageError(f"Redis 会话历史读取失败：{error}") from error

    def append(
        self,
        identity: Identity,
        session_id: str,
        standalone_question: str,
    ) -> None:
        key = conversation_key(identity, session_id, self.key_prefix)
        try:
            with self.client.pipeline(transaction=True) as pipeline:
                pipeline.rpush(key, standalone_question)
                pipeline.ltrim(key, -self.history_turns, -1)
                pipeline.expire(key, self.ttl_seconds)
                pipeline.execute()
        except RedisError as error:
            raise StorageError(f"Redis 会话历史写入失败：{error}") from error


registry.register("conversation_store", "json", JsonConversationStore)
registry.register("conversation_store", "redis", RedisConversationStore)
