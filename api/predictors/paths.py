"""모델 파일 경로를 한 곳에서 정한다. 서빙(predictors)과 학습(training)이 같은 규칙을 쓴다."""
import os


def model_path(filename: str, create_dir: bool = False) -> str:
    """MODEL_DIR(기본 artifacts) 아래 파일 경로를 돌려준다.
    Args:
        filename: 모델·지표 파일 이름 (예: "dl.pt").
        create_dir: True 면 폴더가 없을 때 만든다 (학습 스크립트가 저장할 때).
    Returns:
        파일 경로.
    """
    model_dir = os.getenv("MODEL_DIR", "artifacts")
    if create_dir:
        os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, filename)
