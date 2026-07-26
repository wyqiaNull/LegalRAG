"""为本地演示生成短期 JWT。"""

from __future__ import annotations

import argparse
import uuid
from datetime import timedelta

from legalrag.api.auth import ALL_SCOPES, create_token
from legalrag.config.settings import load_settings
from legalrag.core.models import Confidentiality


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 LegalRAG 演示 JWT")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument(
        "--scope",
        action="append",
        choices=sorted(ALL_SCOPES),
        required=True,
    )
    parser.add_argument(
        "--confidentiality",
        action="append",
        type=Confidentiality,
        choices=list(Confidentiality),
        required=True,
    )
    parser.add_argument("--expires-minutes", type=int, default=60)
    args = parser.parse_args()
    if args.expires_minutes < 1:
        parser.error("--expires-minutes 必须大于 0")

    settings = load_settings()
    configured_roles = {
        policy.role for policy in settings.config.governance.acl_policies
    }
    if args.role not in configured_roles:
        parser.error("--role 不在当前 ACL 配置中")
    print(
        create_token(
            settings,
            user_id=args.user_id,
            role=args.role,
            tenant_id=args.tenant_id,
            allowed_confidentiality=list(dict.fromkeys(args.confidentiality)),
            scopes=set(args.scope),
            token_id=str(uuid.uuid4()),
            expires_in=timedelta(minutes=args.expires_minutes),
        )
    )


if __name__ == "__main__":
    main()
