import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "results"

# Read csv
fedavg = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "baseline_results.csv"
    )
)

fixed = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "fixed_results.csv"
    )
)

adaptive = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "adaptive_results.csv"
    )
)

plt.figure(figsize=(8,5))

# Baseline
plt.plot(
    fedavg["round"],
    fedavg["accuracy"],
    marker="o",
    label="FedAvg"
)

# Fixed
plt.plot(
    fixed["round"],
    fixed["accuracy"],
    marker="s",
    label="Fixed INT8"
)

# Adaptive
plt.plot(
    adaptive["round"],
    adaptive["accuracy"],
    marker="^",
    label="Adaptive"
)

plt.xlabel("Round")

plt.ylabel("Accuracy")

plt.title(
    "Accuracy Comparison Across Rounds"
)

plt.grid(True)

plt.legend()

plt.savefig(
    "accuracy_per_round.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()