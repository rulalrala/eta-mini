import numpy as np
from sklearn.model_selection import train_test_split
from api.predictors import rule

SEED = 0  # 데이터 생성과 학습/평가 분할에 쓰는 시드

def collect_data(n: int = 500, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """배달 기록을 수집한다 (여기서는 생성).

    같은 seed 면 항상 같은 데이터가 나온다. 학습 재현성의 출발점이다.

    Args:
        n: 배달 건수.
        seed: 난수 시드.

    Returns:
        (X, y) 튜플. X 는 [n, 2] (배달 거리 km, 대기 주문 수), y 는 [n] 실제 걸린 시간(분). 모두 float32.
    """
    rng = np.random.default_rng(seed)
    distance_km = rng.uniform(0.5, 8.0, n)
    pending_orders = rng.integers(0, 11, n).astype(float)
    minutes = 12 + 3 * distance_km + 4 * pending_orders + rng.normal(0, 1.5, n)
    X = np.column_stack([distance_km, pending_orders]).astype(np.float32)
    return X, minutes.astype(np.float32)

def load_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """모든 학습 스크립트가 같은 데이터·같은 분할(학습 80% / 평가 20%)을 쓰도록 한 곳에서 만든다.

    Returns:
        (X_tr, X_te, y_tr, y_te).
    """
    X, y = collect_data(seed=SEED)
    return train_test_split(X, y, test_size=0.2, random_state=SEED)

def baseline_mae(X: np.ndarray, y: np.ndarray) -> float:
    """기준선(rule) 구현체의 평균 절대 오차(분). 규칙이 산술식이라 배열을 한 번에 넣어 계산한다.

    Args:
        X: [n, 2] 입력 (배달 거리 km, 대기 주문 수).
        y: [n] 실제 걸린 시간(분).

    Returns:
        평균 절대 오차(분).
    """
    pred = rule.estimate_delivery_minutes(X[:, 0], X[:, 1])
    return float(np.abs(pred - y).mean())
