import json
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from training.data import baseline_mae, load_split
from api.predictors.paths import model_path

X_tr, X_te, y_tr, y_te = load_split()

model = make_pipeline(StandardScaler(), LinearRegression())  # 정규화를 모델 안에
model.fit(X_tr, y_tr)

mae = float(np.abs(model.predict(X_te) - y_te).mean())
baseline = baseline_mae(X_te, y_te)

joblib.dump(model, model_path("sklearn.joblib", create_dir=True))

with open(model_path("metrics_sklearn.json", create_dir=True), "w", encoding="utf-8") as f:
    json.dump({"model": "sklearn", "mae": mae, "baseline_mae": baseline}, f, indent=2)

print(f"sklearn 평균 오차 {mae:.2f}분  (기준선 rule {baseline:.2f}분)")
