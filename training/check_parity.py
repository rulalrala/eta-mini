"""Python API 와 다른 언어 서비스(Go, Rust)의 예측이 같은지 비교한다.

기준은 학습한 원본 모델(MODEL_IMPL=sklearn)로 띄운 Python API 다.
모든 서버가 떠 있어야 하므로 pytest 테스트가 아니라 스크립트로 실행한다. 표준 라이브러리만 쓴다.

    # 기준 서버 (scikit-learn 은 train 그룹에 있다)
    MODEL_IMPL=sklearn uv run --group train uvicorn api.app:app --port 8080
    # 비교
    PY_URL=http://127.0.0.1:8080 \
    TARGETS=go=http://127.0.0.1:8081,rust=http://127.0.0.1:8082 \
    uv run python -m training.check_parity

하나라도 FAIL 이면 exit 1 (CI 가 멈추도록).
"""
import json
import os
import random
import sys
import urllib.error
import urllib.request

TOLERANCE = 1e-4  # 분. 최대 오차가 이보다 작아야 PASS
PY_URL = os.getenv("PY_URL", "http://127.0.0.1:8080")
TARGETS = os.getenv("TARGETS", "go=http://127.0.0.1:8081,rust=http://127.0.0.1:8082")


def parse_targets(spec: str) -> dict[str, str]:
    """'go=http://...,rust=http://...' 를 {이름: URL} 로 바꾼다."""
    targets = {}
    for item in spec.split(","):
        name, sep, url = item.strip().partition("=")
        if not sep or not name or not url:
            sys.exit(f"TARGETS 형식 오류: {item!r} (예: go=http://127.0.0.1:8081)")
        targets[name.strip()] = url.strip().rstrip("/")
    return targets


def make_inputs() -> list[tuple[float, int]]:
    """고정 입력 3건 + seed 0 무작위 20건 (거리 0.5~8km, 주문 0~10건)."""
    rng = random.Random(0)
    fixed = [(3.0, 2), (0.0, 0), (8.0, 10)]
    rand = [(round(rng.uniform(0.5, 8.0), 3), rng.randint(0, 10)) for _ in range(20)]
    return fixed + rand


def predict(base_url: str, distance_km: float, pending_orders: int) -> float:
    """POST /eta 를 호출해 minutes 를 돌려준다."""
    body = json.dumps({"distance_km": distance_km, "pending_orders": pending_orders}).encode()
    req = urllib.request.Request(f"{base_url}/eta", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return float(json.load(resp)["minutes"])


def main() -> int:
    targets = parse_targets(TARGETS)
    inputs = make_inputs()

    try:
        expected = [predict(PY_URL, d, n) for d, n in inputs]
    except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
        print(f"FAIL [python] 기준 서버 {PY_URL} 호출 실패: {e}")
        return 1
    print(f"기준: python {PY_URL}  입력 {len(inputs)}건")

    all_ok = True
    for name, url in targets.items():
        try:
            got = [predict(url, d, n) for d, n in inputs]
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            print(f"FAIL [{name}] {url} 호출 실패: {e}")
            all_ok = False
            continue
        diffs = [abs(g - w) for g, w in zip(got, expected)]
        worst = max(range(len(diffs)), key=diffs.__getitem__)
        ok = diffs[worst] < TOLERANCE
        all_ok &= ok
        d, n = inputs[worst]
        print(f"{'PASS' if ok else 'FAIL'} [{name}] 최대 오차 {diffs[worst]:.2e}분 "
              f"(입력 {d}km/{n}건: python {expected[worst]:.6f}, {name} {got[worst]:.6f})")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
