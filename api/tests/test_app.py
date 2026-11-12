import math

import pytest


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
    """아주 큰 값도 200 과 유한한 예상 시간을 돌려준다."""
    res = client.post("/eta", json={"distance_km": 1_000_000.0, "pending_orders": 100_000})
    assert res.status_code == 200
    minutes = res.json()["minutes"]
    assert math.isfinite(minutes)
    assert minutes == pytest.approx(10 + 3 * 1_000_000 + 2 * 100_000, abs=1e-6)


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
    ],
)
def test_eta_422_points_to_bad_field(client, payload, bad_field):
    """422 응답의 detail 은 문제가 된 필드만 가리킨다."""
    res = client.post("/eta", json=payload)
    assert res.status_code == 422
    fields = {err["loc"][-1] for err in res.json()["detail"]}
    assert fields == {bad_field}


def test_eta_non_json_body_returns_422(client):
    """JSON 이 아닌 본문은 422 로 거부한다."""
    res = client.post("/eta", content="not json", headers={"Content-Type": "application/json"})
    assert res.status_code == 422


def test_eta_get_method_not_allowed(client):
    """/eta 는 POST 전용이며 GET 은 405 를 돌려준다."""
    res = client.get("/eta")
    assert res.status_code == 405
