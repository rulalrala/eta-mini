import importlib
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

MODEL_IMPL = os.getenv("MODEL_IMPL", "rule")
estimate_delivery_minutes = importlib.import_module(f"api.predictors.{MODEL_IMPL}").estimate_delivery_minutes

class EtaRequest(BaseModel):
    """배달 예상 시간 요청"""
    # 모델이 더 많은 정보를 원하면 여기에 필드를 추가
    # 아래 ge=0 와 같은 조건들을 유효성 검사하여, 잘못된 데이터(음수 등)가 들어오는 것을 사전에 방지
    distance_km: float = Field(ge=0, description="가게에서 배달지까지 거리(km)")
    pending_orders: int = Field(ge=0, description="지금 밀려 있는 주문 수")

class EtaResponse(BaseModel):
    """배달 예상 시간 응답"""
    minutes: float
    model: str  # 어떤 구현체가 낸 예측인지 (장애 추적용)

app = FastAPI(title="delivery-eta")


def get_intro_html(model_name: str) -> str:
    """API 소개 페이지 HTML 문서를 생성한다."""
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>배달 예상 시간(ETA) API</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --text: #1e293b;
      --text-muted: #64748b;
      --primary: #2563eb;
      --primary-hover: #1d4ed8;
      --border: #e2e8f0;
      --code-bg: #0f172a;
      --code-text: #e2e8f0;
      --badge-bg: #dbeafe;
      --badge-text: #1e40af;
      --badge-post-bg: #dcfce7;
      --badge-post-text: #166534;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.6;
      padding: 2.5rem 1rem;
    }}
    .container {{
      max-width: 820px;
      margin: 0 auto;
    }}
    header {{
      margin-bottom: 2rem;
      text-align: center;
    }}
    h1 {{
      font-size: 2.25rem;
      font-weight: 700;
      color: #0f172a;
      margin-bottom: 0.5rem;
    }}
    .subtitle {{
      font-size: 1.1rem;
      color: var(--text-muted);
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.75rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    .card-title {{
      font-size: 1.25rem;
      font-weight: 600;
      margin-bottom: 0.75rem;
      color: #0f172a;
    }}
    .endpoint {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem;
      margin-top: 1rem;
    }}
    .endpoint-header {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
      margin-bottom: 0.5rem;
    }}
    .badge {{
      display: inline-block;
      font-size: 0.8rem;
      font-weight: 700;
      padding: 0.2rem 0.6rem;
      border-radius: 4px;
      letter-spacing: 0.5px;
    }}
    .badge-get {{ background: var(--badge-bg); color: var(--badge-text); }}
    .badge-post {{ background: var(--badge-post-bg); color: var(--badge-post-text); }}
    .endpoint-path {{
      font-family: monospace;
      font-size: 1.05rem;
      font-weight: 600;
    }}
    .endpoint-desc {{
      font-size: 0.95rem;
      color: var(--text-muted);
      margin-bottom: 0.5rem;
    }}
    pre {{
      background: var(--code-bg);
      color: var(--code-text);
      padding: 0.75rem 1rem;
      border-radius: 6px;
      font-size: 0.85rem;
      overflow-x: auto;
      margin-top: 0.5rem;
    }}
    .links {{
      display: flex;
      gap: 1rem;
      margin-top: 1rem;
      flex-wrap: wrap;
    }}
    .btn {{
      display: inline-block;
      background: var(--primary);
      color: #ffffff;
      padding: 0.6rem 1.2rem;
      border-radius: 6px;
      text-decoration: none;
      font-weight: 500;
      transition: background 0.15s ease;
    }}
    .btn:hover {{ background: var(--primary-hover); }}
    .btn-secondary {{
      background: #e2e8f0;
      color: #1e293b;
    }}
    .btn-secondary:hover {{ background: #cbd5e1; }}
    footer {{
      text-align: center;
      color: var(--text-muted);
      font-size: 0.85rem;
      margin-top: 2rem;
    }}
    footer .links {{
      justify-content: center;
      margin-top: 0;
      margin-bottom: 1rem;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>배달 예상 시간(ETA) API</h1>
      <p class="subtitle">배달 거리와 대기 주문 수를 기반으로 배달 소요 시간을 예측합니다.</p>
    </header>

    <div class="card">
      <h2 class="card-title">서비스 개요</h2>
      <p>현재 적용된 예측 모델 구현체: <code>{model_name}</code></p>
    </div>

    <div class="card">
      <h2 class="card-title">주요 엔드포인트</h2>

      <div class="endpoint">
        <div class="endpoint-header">
          <span class="badge badge-get">GET</span>
          <span class="endpoint-path">/</span>
        </div>
        <p class="endpoint-desc">API 소개 및 안내 페이지 (HTML)</p>
      </div>

      <div class="endpoint">
        <div class="endpoint-header">
          <span class="badge badge-get">GET</span>
          <span class="endpoint-path">/health</span>
        </div>
        <p class="endpoint-desc">서비스 상태 및 현재 로드된 모델 구현체 확인</p>
        <pre><code>curl -X GET "/health"
# 응답 예시: {{"status": "ok", "model": "{model_name}"}}</code></pre>
      </div>

      <div class="endpoint">
        <div class="endpoint-header">
          <span class="badge badge-post">POST</span>
          <span class="endpoint-path">/eta</span>
        </div>
        <p class="endpoint-desc">거리(km)와 현재 대기 주문 수를 전송하여 예상 소요 시간(분)을 계산</p>
        <pre><code>curl -X POST "/eta" \\
  -H "Content-Type: application/json" \\
  -d '{{"distance_km": 2.5, "pending_orders": 2}}'
# 응답 예시: {{"minutes": 21.5, "model": "{model_name}"}}</code></pre>
      </div>
    </div>

    <footer>
      <div class="links">
        <a class="btn" href="/docs">Swagger API 문서 (/docs)</a>
        <a class="btn btn-secondary" href="/redoc">ReDoc 문서 (/redoc)</a>
      </div>
      <p>&copy; 2026 ETA Service. All rights reserved.</p>
    </footer>
  </div>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """API 소개 페이지(HTML)를 돌려준다."""
    return get_intro_html(MODEL_IMPL)

@app.get("/health")
def health() -> dict[str, str]:
    """서비스 상태와 현재 구현체 이름을 돌려준다."""
    return {"status": "ok", "model": MODEL_IMPL}

@app.post("/eta", response_model=EtaResponse)
def eta(req: EtaRequest) -> EtaResponse:
    """배달 예상 시간을 계산"""
    minutes = estimate_delivery_minutes(req.distance_km, req.pending_orders)
    return EtaResponse(minutes=minutes, model=MODEL_IMPL)
