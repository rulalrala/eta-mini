# Rust ETA API 이미지. 빌드 컨텍스트는 저장소 루트다 (artifacts/, uv.lock 을 써야 하므로).
#   docker build -f api-rust/Dockerfile -t eta-api-rust .
#   docker run --rm -p 8082:8080 eta-api-rust

# ── 1) Rust 빌드 + 계약 테스트 ──
FROM rust:1-trixie AS build
WORKDIR /src
COPY api-rust/ ./
# 캐시 마운트: 의존성 크레이트와 빌드 결과를 다음 빌드에서 재사용한다 (이미지에는 남지 않음)
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo test --locked --release \
 && cargo build --locked --release \
 && cp target/release/eta-api /eta-api

# ── 2) ONNX Runtime 공유 라이브러리 받기 ──
# 버전은 uv.lock 의 onnxruntime 버전을 그대로 쓴다 (Python 쪽과 같은 런타임 → 같은 예측)
FROM debian:trixie-slim AS ort
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
COPY uv.lock /tmp/uv.lock
RUN set -eu; \
    ORT_VERSION=$(awk '$0 == "name = \"onnxruntime\"" { getline; gsub(/^version = "|"$/, ""); print; exit }' /tmp/uv.lock); \
    [ -n "$ORT_VERSION" ] || { echo "uv.lock 에서 onnxruntime 버전을 찾지 못함" >&2; exit 1; }; \
    case "$TARGETARCH" in \
      amd64) ORT_ARCH=x64 ;; \
      arm64) ORT_ARCH=aarch64 ;; \
      *) echo "지원하지 않는 TARGETARCH: $TARGETARCH" >&2; exit 1 ;; \
    esac; \
    echo "ONNX Runtime ${ORT_VERSION} (linux-${ORT_ARCH})"; \
    curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}.tgz" \
      | tar -xz -C /tmp; \
    cp -L "/tmp/onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}/lib/libonnxruntime.so" /opt/libonnxruntime.so

# ── 3) 실행 이미지 ──
FROM debian:trixie-slim
COPY --from=ort /opt/libonnxruntime.so /usr/local/lib/libonnxruntime.so
COPY --from=build /eta-api /usr/local/bin/eta-api
COPY artifacts/model_sklearn.onnx /app/model_sklearn.onnx
ENV PORT=8080 \
    MODEL_PATH=/app/model_sklearn.onnx \
    ORT_LIB_PATH=/usr/local/lib/libonnxruntime.so
USER 65532:65532
EXPOSE 8080
CMD ["eta-api"]
