"""CLI 阶段使用的轻量会话存储。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonConversationStore:
    def __init__(self, path: str, history_turns: int = 4) -> None:
        self.path = Path(path)
        self.history_turns = history_turns

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

    def get(self, session_id: str) -> list[str]:
        return self._load().get(session_id, [])

    def append(self, session_id: str, standalone_question: str) -> None:
        sessions = self._load()
        history = sessions.setdefault(session_id, [])
        history.append(standalone_question)
        sessions[session_id] = history[-self.history_turns :]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.path)
