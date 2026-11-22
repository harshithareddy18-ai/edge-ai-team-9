"""
train.py -- STARTER. Clean the data, fit baselines, train the network, save it for the server.

    python train.py --team 7 --hidden 32,32
Writes model/model.pt and model/config.json, which app/main.py loads.

Look for TODO. The training loop is given; the decisions are yours.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from app.model import FEATURES, BatteryNet

ap = argparse.ArgumentParser()
ap.add_argument("--team", type=int, required=True)
ap.add_argument("--hidden", default="32,32")
ap.add_argument("--epochs", type=int, default=1500)
a = ap.parse_args()
hidden = tuple(int(h) for h in a.hidden.split(","))

df = pd.read_csv(f"data/team_{a.team:02d}_train.csv")
# Identify invalid sensor readings
bad_battery = df["battery_v"] == 0
bad_temperature = df["temp_c"] == -127
bad_wifi = df["wifi_rssi_dbm"] == 0

# Combine all glitch conditions
bad = bad_battery | bad_temperature | bad_wifi

print("Glitch detection results:")
print("Invalid battery readings:", bad_battery.sum())
print("Invalid temperature readings:", bad_temperature.sum())
print("Invalid WiFi readings:", bad_wifi.sum())
print(f"{len(df)} rows, {bad.sum()} glitch rows dropped")

# Keep only valid sensor readings
df = df[~bad].copy()

rng = np.random.default_rng(0)
idx = rng.permutation(len(df)); n_val = len(df) // 5
val, tr = df.iloc[idx[:n_val]], df.iloc[idx[n_val:]]
X, y = tr[FEATURES].to_numpy(np.float32), tr.days_left.to_numpy(np.float32)
Xv, yv = val[FEATURES].to_numpy(np.float32), val.days_left.to_numpy(np.float32)
# Calculate normalization values using training data only
mu = X.mean(axis=0)
sd = X.std(axis=0)

# Avoid division by zero
sd[sd == 0] = 1.0

# Normalize training and validation features
X = (X - mu) / sd
Xv = (Xv - mu) / sd

mae = lambda p, t: float(np.mean(np.abs(p - t)))
print(f"baseline (predict the mean): {mae(np.full_like(yv, y.mean()), yv):.2f} days")
# Linear regression baseline
X_train_linear = np.column_stack([X, np.ones(len(X))])
X_val_linear = np.column_stack([Xv, np.ones(len(Xv))])

weights, _, _, _ = np.linalg.lstsq(
    X_train_linear, y, rcond=None
)

linear_predictions = X_val_linear @ weights

print(
    f"baseline (linear regression): "
    f"{mae(linear_predictions, yv):.2f} days"
)

torch.manual_seed(0)
net = BatteryNet(hidden)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
Xt, yt = torch.tensor(X), torch.tensor(y)
Xvt = torch.tensor(Xv)
for ep in range(a.epochs):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 128):
        b = perm[i:i + 128]
        loss = torch.nn.functional.mse_loss(net(Xt[b]), yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 300 == 0 or ep == a.epochs - 1:
        with torch.no_grad():
            print(f"  epoch {ep:4d}  val MAE {mae(net(Xvt).numpy(), yv):.2f} days")
with torch.no_grad():
    train_predictions = net(Xt).numpy()
    val_predictions = net(Xvt).numpy()

    train_mae = mae(train_predictions, y)
    val_mae = mae(val_predictions, yv)

print(f"network {hidden}:")
print(f"  Training MAE:   {train_mae:.2f} days")
print(f"  Validation MAE: {val_mae:.2f} days")
os.makedirs("model", exist_ok=True)
torch.save(net.state_dict(), "model/model.pt")
json.dump(dict(team=a.team, hidden=list(hidden), mu=mu.tolist(), sd=sd.tolist(),
               features=FEATURES, val_mae=round(val_mae, 2)), open("model/config.json", "w"), indent=2)
print("saved model/model.pt and model/config.json")
