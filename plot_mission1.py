
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Load Team 9 dataset
df = pd.read_csv("data/team_09_train.csv")

# 2. Identify and remove sensor glitches
bad_battery = df["battery_v"] == 0
bad_temperature = df["temp_c"] == -127
bad_wifi = df["wifi_rssi_dbm"] == 0

bad = bad_battery | bad_temperature | bad_wifi
df = df[~bad].copy()

print("Rows remaining after cleaning:", len(df))

# 3. Input features
features = [
    "battery_v",
    "temp_c",
    "rings_per_day",
    "wifi_rssi_dbm",
    "age_months"
]

# 4. Create scatter plots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, feature in enumerate(features):
    axes[i].scatter(
        df[feature],
        df["days_left"],
        alpha=0.5,
        s=15
    )

    axes[i].set_title(f"days_left vs {feature}")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("days_left")
    axes[i].grid(True, alpha=0.3)

# Remove unused sixth subplot
fig.delaxes(axes[5])

fig.suptitle(
    "Team 9 - Days Left vs Input Features",
    fontsize=16
)

plt.tight_layout()

# 5. Save plot
os.makedirs("data", exist_ok=True)
plt.savefig("data/team_09_scatter_plots.png", dpi=200)

print("Scatter plots saved successfully!")
