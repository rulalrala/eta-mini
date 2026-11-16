from functools import cache
import numpy as np
import onnxruntime as ort
from api.predictors.paths import model_path

@cache
def load_model() -> ort.InferenceSession:
    """처음 호출될 때 한 번만 모델 파일을 읽는다. 서버는 시작할 때(lifespan) 미리 호출한다.
    Returns:
        model_sklearn.onnx 를 불러온 ONNX Runtime 세션.
    """
    return ort.InferenceSession(model_path("model_sklearn.onnx"))

def estimate_delivery_minutes(distance_km: float, pending_orders: int) -> float:
    """ONNX 로 변환한 모델로 배달 예상 시간(분)을 계산한다.
    scikit-learn 없이 ONNX Runtime 만으로 실행한다. 정규화는 모델 안에 있다.
    Args:
        distance_km: 배달 거리(km).
        pending_orders: 대기 주문 수.
    Returns:
        예상 시간(분).
    """
    x = np.array([[distance_km, pending_orders]], dtype=np.float32)
    return float(load_model().run(["variable"], {"X": x})[0][0, 0])
