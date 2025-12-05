# SQLite-vec builder stage - separate stage for better caching
FROM python:3.14-slim AS sqlite-vec-builder

# Install build dependencies for compiling sqlite-vec
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    make \
    git \
    gettext \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Build sqlite-vec extension with cache mount for git and build artifacts
RUN --mount=type=cache,target=/var/cache/git \
    --mount=type=cache,target=/tmp/sqlite-vec-build \
    cd /tmp \
    && git clone --depth 1 --branch v0.1.6 https://github.com/asg017/sqlite-vec.git \
    && cd sqlite-vec \
    && make loadable \
    && mkdir -p /sqlite-vec-dist \
    && cp dist/vec0.* /sqlite-vec-dist/

# Main builder stage
FROM python:3.14-slim AS builder

# Create non-root user
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory and change ownership
WORKDIR /app
RUN chown app:app /app

# Switch to non-root user
USER app

# Copy source code and migrations
COPY --chown=app:app src/ ./src/
COPY --chown=app:app migrations/ ./migrations/

RUN --mount=type=cache,target=/home/app/.cache/uv,uid=1000,gid=1000 \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    uv sync --package mcp-optimizer --no-dev --locked --no-editable

# Copy pre-built sqlite-vec extension
COPY --from=sqlite-vec-builder /sqlite-vec-dist/vec0.so /app/.venv/lib/python3.13/site-packages/sqlite_vec/vec0.so
USER root
RUN chown app:app /app/.venv/lib/python3.13/site-packages/sqlite_vec/vec0.so
USER app

# Pre-download fastembed models and tiktoken encodings stage
FROM builder AS model-downloader

# Switch to root to create cache directory, then switch back to app user
USER root
RUN mkdir -p /app/.cache/fastembed /app/.cache/tiktoken && chown -R app:app /app/.cache
USER app

# Set cache directory for fastembed models and tiktoken
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed
ENV TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken

# Pre-download the embedding model by instantiating TextEmbedding
RUN --mount=type=cache,target=/app/.cache/uv,uid=1000,gid=1000 \
    /app/.venv/bin/python -c "\
import os; \
print(f'FASTEMBED_CACHE_PATH: {os.environ.get(\"FASTEMBED_CACHE_PATH\")}'); \
from fastembed import TextEmbedding; \
print('Downloading embedding model...'); \
model = TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); \
print('Model downloaded successfully')"

# Pre-download tiktoken encodings for offline use
RUN /app/.venv/bin/python -c "\
import tiktoken; \
print('Downloading tiktoken encodings...'); \
tiktoken.get_encoding('cl100k_base'); \
print('Tiktoken encodings downloaded successfully')"

FROM python:3.14-slim AS runner

# Create non-root user (same as builder stage)
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app

# Create app directory and set ownership
WORKDIR /app
RUN chown app:app /app

# Copy the environment and migrations
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/migrations /app/migrations

# Copy pre-downloaded fastembed models and tiktoken encodings
COPY --from=model-downloader --chown=app:app /app/.cache/fastembed /app/.cache/fastembed
COPY --from=model-downloader --chown=app:app /app/.cache/tiktoken /app/.cache/tiktoken

# Switch to non-root user
USER app

# Set default environment variables for container deployment
ENV TOOLHIVE_HOST=host.docker.internal
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed
ENV TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken
ENV COLORED_LOGS=false

# Run the application
CMD ["/app/.venv/bin/mcpo"]
