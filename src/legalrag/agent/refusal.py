"""确定性拒答原因与响应装配。"""

from __future__ import annotations

from enum import Enum

from ..core.models import Answer, Route


class RefusalReason(str, Enum):
    """可评测、可复用的拒答原因码。"""

    OUT_OF_SCOPE = "out_of_scope"
    INSUFFICIENT_INFORMATION = "insufficient_information"
    NO_ACCESSIBLE_CONTEXT = "no_accessible_context"
    CONTRACT_REVIEW_UNAVAILABLE = "contract_review_unavailable"


_REFUSAL_TEXT: dict[RefusalReason, str] = {
    RefusalReason.OUT_OF_SCOPE: "当前知识库未覆盖该问题，无法提供可靠回答。",
    RefusalReason.INSUFFICIENT_INFORMATION: (
        "现有信息不足，无法提供可靠回答。请补充具体法规、制度、合同或问题背景。"
    ),
    RefusalReason.NO_ACCESSIBLE_CONTEXT: (
        "未检索到您有权访问且足以支持回答的资料，无法提供可靠回答。"
    ),
    RefusalReason.CONTRACT_REVIEW_UNAVAILABLE: (
        "已识别为合同审查请求，当前版本尚未开放合同审查功能。"
    ),
}


class RefusalResponder:
    """装配不依赖 LLM 的统一拒答，避免拒答路径继续生成内容。"""

    def refuse(self, reason: RefusalReason, route: Route = Route.NORMAL) -> Answer:
        return Answer(
            text=_REFUSAL_TEXT[reason],
            refused=True,
            reason=reason.value,
            route=route,
        )
