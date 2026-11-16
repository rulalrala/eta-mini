import math

import pytest

from api.app import MAX_DISTANCE_KM, MAX_PENDING_ORDERS, load_predictor


# ---------- / (루트 소개 페이지) ----------

def test_root_returns_html_and_status_200(client):
    """/ 는 200 과 HTML 응답(text/html)을 돌려준다."""
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")


def test_root_contains_key_service_info(client):
    """/ 응답 본문은 서비스 제목과 주요 엔드포인트(/eta, /health, /docs)를 포함한다."""
    res = client.get("/")
    assert res.status_code == 200
    content = res.text
    assert "배달 예상 시간(ETA) API" in content
    assert "/health" in content
    assert "/eta" in content
    assert "/docs" in content
    assert "/redoc" in content


def test_root_contains_current_model(client):
    """/ 응답 본문은 현재 활성화된 모델 구현체 이름을 포함한다."""
    health_model = client.get("/health").json()["model"]
    res = client.get("/")
    assert res.status_code == 200
    assert health_model in res.text


def test_root_post_method_not_allowed(client):
    """/ 는 GET 전용이며 POST 는 405 를 돌려준다."""
    res = client.post("/")
    assert res.status_code == 405


# ---------- /health ----------

def test_health_returns_ok_and_model(client):
    """/health 는 200 과 status=ok, 현재 구현체 이름(rule)을 돌려준다."""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "model": "rule"}


# ---------- /eta 정상 입력 ----------

def test_eta_response_shape(client):
    """/eta 응답은 minutes(숫자)와 model(문자열) 두 필드만 가진다."""
    res = client.post("/eta", json={"distance_km": 2.0, "pending_orders": 1})
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"minutes", "model"}
    assert isinstance(body["minutes"], (int, float))
    assert isinstance(body["model"], str)


def test_eta_model_matches_health(client):
    """/eta 의 model 은 /health 가 알려 주는 구현체 이름과 같다."""
    health_model = client.get("/health").json()["model"]
    res = client.post("/eta", json={"distance_km": 1.0, "pending_orders": 0})
    assert res.json()["model"] == health_model


@pytest.mark.parametrize(
    ("distance_km", "pending_orders", "expected"),
    [
        (2.0, 1, 18.0),   # 10 + 3*2 + 2*1
        (5.0, 3, 31.0),   # 10 + 3*5 + 2*3
        (1.5, 0, 14.5),   # 소수 거리
        (0.0, 4, 18.0),   # 거리 0, 주문만 있음
    ],
)
def test_eta_known_values(client, distance_km, pending_orders, expected):
    """알려진 입력에 대해 기준선 규칙의 예상 시간을 돌려준다."""
    res = client.post("/eta", json={"distance_km": distance_km, "pending_orders": pending_orders})
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(expected)


def test_eta_accepts_integer_distance(client):
    """distance_km 에 정수를 보내도 받아들인다."""
    res = client.post("/eta", json={"distance_km": 3, "pending_orders": 0})
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(19.0)


# ---------- /eta 경계값 ----------

def test_eta_zero_inputs(client):
    """거리 0, 대기 주문 0 은 허용되며 기본 시간 10분을 돌려준다."""
    res = client.post("/eta", json={"distance_km": 0, "pending_orders": 0})
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(10.0)


def test_eta_large_inputs(client):
    """상한값(거리 1,000,000km, 주문 1,000,000건)은 허용되며 200 과 유한한 예상 시간을 돌려준다."""
    res = client.post("/eta", json={"distance_km": MAX_DISTANCE_KM, "pending_orders": MAX_PENDING_ORDERS})
    assert res.status_code == 200
    minutes = res.json()["minutes"]
    assert math.isfinite(minutes)
    assert minutes == pytest.approx(10 + 3 * 1_000_000 + 2 * 1_000_000, abs=1e-6)


def test_eta_increases_with_inputs(client):
    """거리나 대기 주문이 늘면 예상 시간이 줄지 않는다."""
    def minutes(d, p):
        return client.post("/eta", json={"distance_km": d, "pending_orders": p}).json()["minutes"]

    assert minutes(5.0, 0) > minutes(1.0, 0)
    assert minutes(1.0, 5) > minutes(1.0, 0)


# ---------- /eta 잘못된 입력 → 422 ----------

@pytest.mark.parametrize(
    "payload",
    [
        {"distance_km": -1.0, "pending_orders": 0},     # 음수 거리
        {"distance_km": 1.0, "pending_orders": -1},     # 음수 주문 수
        {"distance_km": -0.001, "pending_orders": 0},   # 0 바로 아래
        {"distance_km": "far", "pending_orders": 0},    # 숫자가 아닌 거리
        {"distance_km": 1.0, "pending_orders": "many"}, # 숫자가 아닌 주문 수
        {"distance_km": 1.0, "pending_orders": 1.5},    # 정수가 아닌 주문 수
        {"distance_km": None, "pending_orders": 0},     # null
        {"distance_km": "inf", "pending_orders": 0},    # 무한대 문자열
        {"distance_km": "Infinity", "pending_orders": 0},  # 무한대 문자열
        {"distance_km": "nan", "pending_orders": 0},    # NaN 문자열
        {"distance_km": 1_000_000.5, "pending_orders": 0},  # 거리 상한 바로 위
        {"distance_km": 1.0, "pending_orders": 1_000_001},  # 주문 수 상한 바로 위
        {"distance_km": 1.0, "pending_orders": 10**400},    # 아주 큰 정수 (이전에는 500)
        {"pending_orders": 0},                          # 거리 누락
        {"distance_km": 1.0},                           # 주문 수 누락
        {},                                             # 빈 본문
    ],
)
def test_eta_invalid_input_returns_422(client, payload):
    """음수·숫자 아님·누락 등 잘못된 입력은 422 로 거부한다."""
    res = client.post("/eta", json=payload)
    assert res.status_code == 422


@pytest.mark.parametrize(
    ("payload", "bad_field"),
    [
        ({"distance_km": -1.0, "pending_orders": 0}, "distance_km"),
        ({"distance_km": 1.0, "pending_orders": -1}, "pending_orders"),
        ({"distance_km": "far", "pending_orders": 0}, "distance_km"),
        ({"pending_orders": 0}, "distance_km"),
        ({"distance_km": 1.0}, "pending_orders"),
        ({"distance_km": "inf", "pending_orders": 0}, "distance_km"),
        ({"distance_km": 1.0, "pending_orders": 10**400}, "pending_orders"),
    ],
)
def test_eta_422_points_to_bad_field(client, payload, bad_field):
    """422 응답의 detail 은 문제가 된 필드만 가리킨다."""
    res = client.post("/eta", json=payload)
    assert res.status_code == 422
    fields = {err["loc"][-1] for err in res.json()["detail"]}
    assert fields == {bad_field}


@pytest.mark.parametrize("value", ["inf", "-inf", "Infinity", "nan"])
def test_eta_non_finite_distance_error_type(client, value):
    """무한대·NaN 거리는 422 이고 오류 type 이 finite_number 다 (Go·Rust 구현이 따르는 계약)."""
    res = client.post("/eta", json={"distance_km": value, "pending_orders": 0})
    assert res.status_code == 422
    assert [err["type"] for err in res.json()["detail"]] == ["finite_number"]


def test_eta_overflowing_number_returns_422(client):
    """float 범위를 넘는 숫자(1e309 → 무한대)는 minutes=null 대신 422 로 거부한다."""
    res = client.post("/eta", content='{"distance_km": 1e309, "pending_orders": 0}',
                      headers={"Content-Type": "application/json"})
    assert res.status_code == 422


def test_eta_non_json_body_returns_422(client):
    """JSON 이 아닌 본문은 422 로 거부한다."""
    res = client.post("/eta", content="not json", headers={"Content-Type": "application/json"})
    assert res.status_code == 422


def test_eta_get_method_not_allowed(client):
    """/eta 는 POST 전용이며 GET 은 405 를 돌려준다."""
    res = client.get("/eta")
    assert res.status_code == 405


# ---------- MODEL_IMPL ----------

@pytest.mark.parametrize("name", ["nope", "paths", "../rule", ""])
def test_unknown_model_impl_is_rejected(name):
    """목록에 없는 MODEL_IMPL 은 모듈을 import 하지 않고 바로 RuntimeError 로 거부한다."""
    with pytest.raises(RuntimeError, match="MODEL_IMPL"):
        load_predictor(name)


def test_known_model_impl_loads():
    """rule 구현체는 estimate_delivery_minutes 를 가진 모듈로 불러와진다."""
    assert callable(load_predictor("rule").estimate_delivery_minutes)
