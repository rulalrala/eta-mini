"""PyTorch 학습: 배달 기록 수집 → 분할 → 정규화 통계 계산 → 학습 루프 → 평가 → 저장."""
import json
import numpy as np
import torch
from training.data import baseline_mae, load_split
from api.predictors.dl import build_model
from api.predictors.paths import model_path

# 실험 추적 대상: 이 값에 따라 결과가 달라진다
HPARAMS = {"lr": 0.5, "epochs": 1000, "seed": 0}

torch.manual_seed(HPARAMS["seed"])
X_tr, X_te, y_tr, y_te = load_split()

model = build_model()
# 정규화 통계는 학습 데이터에서만 계산
model.mean.copy_(torch.tensor(np.mean(X_tr, axis=0)))
model.std.copy_(torch.tensor(np.std(X_tr, axis=0)))
Xt, yt = torch.tensor(X_tr), torch.tensor(y_tr).unsqueeze(1)
opt = torch.optim.Adam(model.parameters(), lr=HPARAMS["lr"])

for _ in range(HPARAMS["epochs"]):
    opt.zero_grad()
    loss = torch.nn.functional.mse_loss(model(Xt), yt)
    loss.backward()
    opt.step()

with torch.no_grad():
    pred = model(torch.tensor(X_te)).squeeze(1).numpy()

mae = float(np.abs(pred - y_te).mean())
baseline = baseline_mae(X_te, y_te)
torch.save(model.state_dict(), model_path("dl.pt", create_dir=True))

with open(model_path("metrics_dl.json", create_dir=True), "w", encoding="utf-8") as f:
    json.dump({"model": "dl", "mae": mae, "baseline_mae": baseline, "hparams": HPARAMS}, f, indent=2)
print(f"dl 평균 오차 {mae:.2f}분  (기준선 rule {baseline:.2f}분)  hparams={HPARAMS}")
