import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "results"

adaptive = pd.read_csv(

    os.path.join(

        BASE_DIR,

        "adaptive_results.csv"

    )

)

plt.figure(

    figsize=(8,5)

)

plt.plot(

    adaptive["round"],

    adaptive["avg_bit_width"],

    marker="o"

)

plt.xlabel(

    "Round"

)

plt.ylabel(

    "Average Bit-width"

)

plt.title(

    "Adaptive Bit-width Selection Across Rounds"

)

plt.grid(True)

plt.savefig(

    "bitwidth_per_round.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()