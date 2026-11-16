# Go ETA API 이미지. 빌드 컨텍스트는 저장소 루트다 (artifacts/, uv.lock 을 써야 하므로).
#   docker build -f api-go/Dockerfile -t eta-api-go .
#   docker run --rm -p 8081:8080 eta-api-go

# ── 1) Go 빌드 + 계약 테스트 ──
# onnxruntime_go 는 cgo 로 ONNX Runtime 을 불러오므로 CGO_ENABLED=1 (glibc 기반 이미지)
FROM golang:1.25-trixie AS build
WORKDIR /src
COPY api-go/go.mod api-go/go.sum ./
RUN go mod download
COPY api-go/ ./
RUN go vet ./... && go test ./... \
 && CGO_ENABLED=1 go build -trimpath -ldflags="-s -w" -o /out/eta-api .

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
COPY --from=build /out/eta-api /usr/local/bin/eta-api
COPY artifacts/model_sklearn.onnx /app/model_sklearn.onnx
ENV PORT=8080 \
    MODEL_PATH=/app/model_sklearn.onnx \
    ORT_LIB_PATH=/usr/local/lib/libonnxruntime.so
USER 65532:65532
EXPOSE 8080
CMD ["eta-api"]
