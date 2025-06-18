FROM cgr.dev/chainguard/wolfi-base:latest AS base
ENV LANG="C.UTF-8"
ENV UV_FROZEN=true
ENV UV_NO_CACHE=true

WORKDIR /app
RUN chmod 777 /app
RUN apk add --no-cache wget bash tzdata
RUN apk add --no-cache python-3.12-dev
RUN apk add --no-cache uv
USER root

COPY ./uv.lock ./pyproject.toml ./
RUN apk add --no-cache --virtual core-py-build-deps git \
  && uv sync --no-editable --no-install-package mostlyai --no-install-project \
  && apk del core-py-build-deps git

COPY mostlyai ./mostlyai
COPY README.md ./
RUN uv sync --no-editable

USER nonroot

EXPOSE 8080
ENTRYPOINT [ "/app/.venv/bin/mostlyai-mcp-server", "--host", "0.0.0.0", "--port", "8000", "--transport", "sse" ]
