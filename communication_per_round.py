import os
import pandas as pd
import matplotlib.pyplot as plt

# ====== 自动定位 results 文件夹 ======
BASE_DIR = "results"

fedavg = pd.read_csv(os.path.join(BASE_DIR, "baseline_results.csv"))
fixed = pd.read_csv(os.path.join(BASE_DIR, "fixed_results.csv"))
adaptive = pd.read_csv(os.path.join(BASE_DIR, "adaptive_results.csv"))

# ====== 画图 ======
plt.figure(figsize=(8,5))

plt.plot(fedavg["round"], fedavg["total_comm_kb"], label="FedAvg", marker="o")
plt.plot(fixed["round"], fixed["total_comm_kb"], label="Fixed INT8", marker="s")
plt.plot(adaptive["round"], adaptive["total_comm_kb"], label="Adaptive", marker="^")

plt.xlabel("Round")
plt.ylabel("Total Communication (KB)")
plt.title("Per-round Communication Cost Comparison")
plt.legend()
plt.grid(True)

# ====== 保存（论文必须） ======
plt.savefig("communication_per_round.png", dpi=300, bbox_inches="tight")

plt.show()