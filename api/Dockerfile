FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-cache
ENV PATH="/app/.venv/bin:$PATH"
COPY api/ api/
ARG MODEL_IMPL=rule
ENV MODEL_IMPL=${MODEL_IMPL}
EXPOSE 8081
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8081"]

# onnx 구현체가 읽을 모델 파일 (train Job 이 만든 것)
COPY artifacts/model_sklearn.onnx artifacts/