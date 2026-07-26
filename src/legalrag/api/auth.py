"""JWT 验证与 API scope 授权。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import InvalidTokenError
from pydantic import BaseModel, Field, ValidationError

from ..config.settings import Settings
from ..core.models import Confidentiality, Identity

ALL_SCOPES = {
    "query",
    "ingest",
    "contract:review",
    "feedback",
    "documents:read",
    "documents:delete",
    "admin:global",
}


class Principal(BaseModel):
    user_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    allowed_confidentiality: list[Confidentiality] = Field(min_length=1)
    scopes: set[str]
    token_id: str = Field(min_length=1)

    def identity(self) -> Identity:
        return Identity(
            user_id=self.user_id,
            role=self.role,
            tenant_id=self.tenant_id,
            allowed_confidentiality=self.allowed_confidentiality,
        )


_bearer = HTTPBearer(auto_error=False)


def _unauthorized(detail: str = "无效或缺失的访问令牌") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def decode_token(token: str, settings: Settings) -> Principal:
    secrets = settings.secrets
    if not secrets.jwt_secret_key:
        raise HTTPException(status_code=503, detail="JWT 服务未配置")
    if len(secrets.jwt_secret_key.encode()) < 32:
        raise HTTPException(status_code=503, detail="JWT 密钥长度不足")
    try:
        claims = jwt.decode(
            token,
            secrets.jwt_secret_key,
            algorithms=[settings.config.service.jwt_algorithm],
            issuer=secrets.jwt_issuer,
            audience=secrets.jwt_audience,
            options={
                "require": [
                    "sub",
                    "role",
                    "tenant_id",
                    "allowed_confidentiality",
                    "scope",
                    "iss",
                    "aud",
                    "iat",
                    "exp",
                    "jti",
                ]
            },
        )
        raw_scope = claims["scope"]
        scopes = set(raw_scope.split()) if isinstance(raw_scope, str) else set(raw_scope)
        if not scopes <= ALL_SCOPES:
            raise ValueError("令牌包含未知 scope")
        principal = Principal(
            user_id=claims["sub"],
            role=claims["role"],
            tenant_id=claims["tenant_id"],
            allowed_confidentiality=claims["allowed_confidentiality"],
            scopes=scopes,
            token_id=claims["jti"],
        )
        configured_roles = {
            policy.role for policy in settings.config.governance.acl_policies
        }
        if principal.role not in configured_roles:
            raise ValueError("令牌角色未配置 ACL")
        return principal
    except (InvalidTokenError, KeyError, TypeError, ValueError, ValidationError) as exc:
        raise _unauthorized() from exc


def create_token(
    settings: Settings,
    *,
    user_id: str,
    role: str,
    tenant_id: str,
    allowed_confidentiality: list[Confidentiality],
    scopes: set[str],
    token_id: str,
    expires_in: timedelta = timedelta(hours=1),
) -> str:
    if len(settings.secrets.jwt_secret_key.encode()) < 32:
        raise ValueError("JWT_SECRET_KEY 必须至少为 32 字节")
    if not scopes <= ALL_SCOPES:
        raise ValueError("包含未知 scope")
    now = datetime.now(UTC)
    claims: dict[str, Any] = {
        "sub": user_id,
        "role": role,
        "tenant_id": tenant_id,
        "allowed_confidentiality": [value.value for value in allowed_confidentiality],
        "scope": " ".join(sorted(scopes)),
        "iss": settings.secrets.jwt_issuer,
        "aud": settings.secrets.jwt_audience,
        "iat": now,
        "exp": now + expires_in,
        "jti": token_id,
    }
    return jwt.encode(
        claims,
        settings.secrets.jwt_secret_key,
        algorithm=settings.config.service.jwt_algorithm,
    )


def principal_dependency(settings_dependency):
    async def get_principal(
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
        settings: Settings = Depends(settings_dependency),
    ) -> Principal:
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise _unauthorized()
        return decode_token(credentials.credentials, settings)

    return get_principal


def require_scopes(principal_dependency_callable, *required: str):
    async def dependency(
        principal: Principal = Depends(principal_dependency_callable),
    ) -> Principal:
        if not set(required) <= principal.scopes:
            raise HTTPException(status_code=403, detail="令牌缺少所需权限")
        return principal

    return dependency
