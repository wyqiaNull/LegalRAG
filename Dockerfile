FROM ghcr.io/astral-sh/uv:0.8.22 AS uv

FROM python:3.11-slim AS runtime-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

COPY --from=uv /uv /uvx /bin/

RUN groupadd --system legalrag \
    && useradd --system --gid legalrag --home-dir /app legalrag

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra service --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra service

COPY config ./config
COPY scripts/create_jwt.py scripts/compose_smoke.py ./scripts/
RUN mkdir -p /var/lib/legalrag/uploads \
    && chown -R legalrag:legalrag /var/lib/legalrag

USER legalrag
EXPOSE 8000

CMD ["uvicorn", "legalrag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime-base AS test

USER root
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra service --extra dev
COPY tests ./tests
COPY data/eval ./data/eval
RUN chown -R legalrag:legalrag /app/tests /app/data/eval
USER legalrag
CMD ["pytest", "-q", "-o", "cache_dir=/tmp/pytest-cache"]

FROM runtime-base AS runtime
