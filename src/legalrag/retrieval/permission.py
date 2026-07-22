"""基于 ACL 的检索权限预过滤。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ValidationError

from ..core import registry
from ..core.errors import ConfigError
from ..core.interfaces import Filter, MetadataStore, PermissionFilter
from ..core.models import ROLE_WILDCARD, Confidentiality, DocType, Identity

_DENY_ALL: Filter = {"chunk_id": []}


class AclPolicy(BaseModel):
    role: str
    allowed_confidentiality: list[Confidentiality]
    allowed_doc_types: list[DocType]
    tenant_scope: Literal["own"]


class AclPermissionFilter(PermissionFilter):
    def __init__(
        self,
        metadata_store: MetadataStore,
        shared_tenant_id: str = "__global__",
        shared_doc_types: list[DocType] | None = None,
    ) -> None:
        self.metadata_store = metadata_store
        self.shared_tenant_id = shared_tenant_id
        self.shared_doc_types = [
            DocType(doc_type).value
            for doc_type in (
                [DocType.REGULATION]
                if shared_doc_types is None
                else shared_doc_types
            )
        ]

    def build(self, identity: Identity) -> Filter:
        raw_policy = self.metadata_store.get_acl(identity.role)
        if raw_policy is None:
            return dict(_DENY_ALL)
        try:
            policy = AclPolicy.model_validate(raw_policy)
        except ValidationError as exc:
            raise ConfigError(f"角色 {identity.role!r} 的 ACL 策略无效") from exc
        if policy.role != identity.role:
            raise ConfigError(f"角色 {identity.role!r} 的 ACL 策略不匹配")

        requested = set(identity.allowed_confidentiality)
        confidentiality = [
            level.value
            for level in policy.allowed_confidentiality
            if level in requested
        ]
        doc_types = [doc_type.value for doc_type in policy.allowed_doc_types]
        if (
            not identity.tenant_id
            or identity.tenant_id == self.shared_tenant_id
            or not confidentiality
            or not doc_types
        ):
            return dict(_DENY_ALL)

        tenant_branches: list[Filter] = [{"tenant_id": identity.tenant_id}]
        if self.shared_doc_types:
            tenant_branches.append(
                {
                    "tenant_id": self.shared_tenant_id,
                    "confidentiality": [Confidentiality.PUBLIC.value],
                    "doc_type": self.shared_doc_types,
                    "allowed_roles": [ROLE_WILDCARD],
                }
            )
        return {
            "$or": tenant_branches,
            "confidentiality": confidentiality,
            "doc_type": doc_types,
            "allowed_roles": [identity.role, ROLE_WILDCARD],
        }


registry.register("permission_filter", "acl", AclPermissionFilter)
