
import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from app.model import FEATURES, BatteryNet


# =====================================================
# 1. LOAD AND CLEAN TEAM 9 DATASET
# =====================================================

df = pd.read_csv("data/team_09_train.csv")

bad_battery = df["battery_v"] == 0
bad_temperature = df["temp_c"] == -127
bad_wifi = df["wifi_rssi_dbm"] == 0

bad = bad_battery | bad_temperature | bad_wifi

df = df[~bad].copy()

print("Clean dataset rows:", len(df))


# =====================================================
# 2. RECREATE THE ORIGINAL VALIDATION SPLIT
# =====================================================

rng = np.random.default_rng(0)

idx = rng.permutation(len(df))
n_val = len(df) // 5

val = df.iloc[idx[:n_val]]

Xv = val[FEATURES].to_numpy(np.float32)
yv = val["days_left"].to_numpy(np.float32)


# =====================================================
# 3. LOAD SAVED MODEL AND NORMALIZATION VALUES
# =====================================================

with open("model/config.json", "r") as f:
    cfg = json.load(f)

mu = np.array(cfg["mu"], dtype=np.float32)
sd = np.array(cfg["sd"], dtype=np.float32)

Xv_norm = (Xv - mu) / sd

net = BatteryNet(tuple(cfg["hidden"]))

net.load_state_dict(
    torch.load(
        "model/model.pt",
        map_location="cpu",
        weights_only=True
    )
)

net.eval()

with torch.no_grad():
    predictions = net(
        torch.from_numpy(Xv_norm)
    ).numpy()

errors = np.abs(predictions - yv)

print("Validation MAE:", round(float(errors.mean()), 2), "days")
print("Largest prediction error:", round(float(errors.max()), 2), "days")


# =====================================================
# 4. PREDICTED VS ACTUAL PLOT
# =====================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    yv,
    predictions,
    alpha=0.55,
    label="Predictions"
)

# Highlight 10 largest prediction errors
largest = np.argsort(errors)[-10:]

plt.scatter(
    yv[largest],
    predictions[largest],
    color="red",
    s=65,
    label="10 largest errors"
)

# Perfect prediction reference line
minimum = min(yv.min(), predictions.min())
maximum = max(yv.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "k--",
    label="Perfect prediction"
)

plt.xlabel("Actual days_left")
plt.ylabel("Predicted days_left")

plt.title(
    "Team 9 - Predicted vs Actual Battery Life"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

os.makedirs("data", exist_ok=True)

plt.savefig(
    "data/team_09_predicted_vs_actual.png",
    dpi=200
)

plt.close()


# =====================================================
# 5. ARCHITECTURE EXPERIMENT RESULTS
# =====================================================

architectures = [
    "(4,)",
    "(8,)",
    "(16,)",
    "(32,)",
    "(64,)",
    "(4,4)",
    "(8,8)",
    "(16,16)",
    "(32,32)",
    "(64,64)"
]

val_maes = [
    9.36,
    13.09,
    9.06,
    7.12,
    6.17,
    6.70,
    4.09,
    3.77,
    3.46,
    3.50
]

# Correct parameter counts:
# Single hidden layer: 7h + 1
# Two equal hidden layers: h*h + 8h + 1

parameter_counts = [
    7 * 4 + 1,
    7 * 8 + 1,
    7 * 16 + 1,
    7 * 32 + 1,
    7 * 64 + 1,
    4**2 + 8 * 4 + 1,
    8**2 + 8 * 8 + 1,
    16**2 + 8 * 16 + 1,
    32**2 + 8 * 32 + 1,
    64**2 + 8 * 64 + 1
]


# =====================================================
# 6. ARCHITECTURE COMPARISON PLOT
# =====================================================

plt.figure(figsize=(11, 7))

# Single hidden-layer models
plt.scatter(
    parameter_counts[:5],
    val_maes[:5],
    marker="o",
    s=85,
    label="Single hidden layer"
)

# Two hidden-layer models
plt.scatter(
    parameter_counts[5:],
    val_maes[5:],
    marker="s",
    s=85,
    label="Two hidden layers"
)

# Highlight selected architecture (32,32)
selected_index = 8

plt.scatter(
    parameter_counts[selected_index],
    val_maes[selected_index],
    marker="*",
    s=300,
    label="Selected model (32,32)"
)

# Add architecture labels
for i, name in enumerate(architectures):

    plt.annotate(
        name,
        (
            parameter_counts[i],
            val_maes[i]
        ),
        xytext=(6, 7),
        textcoords="offset points",
        fontsize=9
    )

plt.xscale("log")

plt.xlabel("Number of parameters (log scale)")
plt.ylabel("Validation MAE (days)")

plt.title(
    "Team 9 - Architecture Comparison"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "data/team_09_architecture_comparison.png",
    dpi=200
)

plt.close()


# =====================================================
# 7. FINAL OUTPUT
# =====================================================

print("\nMission 2 plots generated successfully!")

print(
    "1. data/team_09_predicted_vs_actual.png"
)

print(
    "2. data/team_09_architecture_comparison.png"
)

print("\nSelected architecture: (32,32)")
print("Parameters: 1281")
print("Validation MAE: 3.46 days")


