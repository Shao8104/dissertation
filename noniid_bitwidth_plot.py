import os
import pandas as pd
import matplotlib.pyplot as plt

RESULT_DIR = "results"
FIGURE_DIR = os.path.join(RESULT_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

adaptive = pd.read_csv(
    os.path.join(
        RESULT_DIR,
        "adaptive_noniid_results.csv"
    )
)

plt.figure(figsize=(10, 6))

plt.plot(
    adaptive["round"],
    adaptive["avg_bit_width"],
    marker="o",
    linewidth=2
)

plt.xlabel("Round")
plt.ylabel("Average Bit-width")
plt.title("Adaptive Bit-width Selection under Non-IID")

plt.grid(True)

plt.tight_layout()

save_path = os.path.join(
    FIGURE_DIR,
    "bitwidth_noniid.png"
)

plt.savefig(save_path, dpi=300)
plt.show()

print(f"Saved: {save_path}")