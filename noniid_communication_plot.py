import os
import pandas as pd
import matplotlib.pyplot as plt

RESULT_DIR = "results"
FIGURE_DIR = os.path.join(RESULT_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

baseline = pd.read_csv(
    os.path.join(
        RESULT_DIR,
        "baseline_noniid_results.csv"
    )
)

fixed = pd.read_csv(
    os.path.join(
        RESULT_DIR,
        "fixed_noniid_results.csv"
    )
)

adaptive = pd.read_csv(
    os.path.join(
        RESULT_DIR,
        "adaptive_noniid_results.csv"
    )
)


plt.figure(figsize=(10, 6))

plt.plot(
    baseline["round"],
    baseline["total_comm_kb"],
    marker="o",
    linewidth=2,
    label="FedAvg"
)

plt.plot(
    fixed["round"],
    fixed["total_comm_kb"],
    marker="s",
    linewidth=2,
    label="Fixed INT8"
)

plt.plot(
    adaptive["round"],
    adaptive["total_comm_kb"],
    marker="^",
    linewidth=2,
    label="Adaptive"
)

plt.xlabel("Round")
plt.ylabel("Total Communication (KB)")
plt.title("Communication Cost Comparison under Non-IID")
plt.legend()
plt.grid(True)

plt.tight_layout()

save_path = os.path.join(
    FIGURE_DIR,
    "communication_noniid.png"
)

plt.savefig(save_path, dpi=300)
plt.show()
print(f"Saved: {save_path}")