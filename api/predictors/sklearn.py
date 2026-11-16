import os
from functools import cache
import joblib
import numpy as np

@cache
def load_model():
    """처음 호출될 때 한 번만 모델 파일을 읽는다 (지연 로딩).
    Returns:
        학습 스크립트가 저장한 scikit-learn 파이프라인 (StandardScaler → LinearRegression).
    """
    return joblib.load(os.path.join(os.getenv("MODEL_DIR", "artifacts"), "sklearn.joblib"))

def estimate_delivery_minutes(distance_km: float, pending_orders: int) -> float:
    """학습된 scikit-learn 모델(원본)로 배달 예상 시간(분)을 계산한다.
    다른 구현체(ONNX, Go, Rust)의 예측이 맞는지 비교할 때 기준이 된다. 정규화는 파이프라인 안에 있다.
    scikit-learn·joblib 은 train 그룹이므로 `uv run --group train` 으로 띄운다.
    Args:
        distance_km: 배달 거리(km).
        pending_orders: 대기 주문 수.
    Returns:
        예상 시간(분).
    """
    x = np.array([[distance_km, pending_orders]], dtype=np.float64)
    return float(load_model().predict(x)[0])
