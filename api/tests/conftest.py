import os

# 예상값 테스트는 기준선(rule) 구현체를 전제로 한다.
# app 은 import 시점에 MODEL_IMPL 을 읽으므로, import 전에 고정한다.
os.environ["MODEL_IMPL"] = "rule"

import pytest
from fastapi.testclient import TestClient

from api.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
