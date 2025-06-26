FROM harbor.m-dev.mostlylab.com/wolfi/python:312 AS base
ENV UV_FROZEN=true
ENV UV_NO_CACHE=true

USER root
COPY --chmod=644 ./uv.lock ./pyproject.toml ./
RUN apk add --no-cache --virtual temp-build-deps git \
  && uv sync --no-editable --no-install-package mostlyai --no-install-project \
  && apk del temp-build-deps git

COPY --chmod=644 mostlyai ./mostlyai
COPY --chmod=644 README.md ./
RUN uv sync --no-editable

USER nonroot
ENV FASTMCP_PORT=8080
ENV FASTMCP_HOST=0.0.0.0

ARG GIT_COMMIT_SHA=unknown
LABEL GIT_COMMIT_SHA=$GIT_COMMIT_SHA
ENV GIT_COMMIT_SHA=${GIT_COMMIT_SHA}

EXPOSE ${FASTMCP_PORT}
ENTRYPOINT [ "/app/.venv/bin/mostlyai-mcp-server", "--transport", "sse"]
