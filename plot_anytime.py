"""
=============================================================================
Generate the anytime-performance plot (Figure 3) from anytime_curves.csv.

anytime_curves.csv is produced by hpo_benchmark_v2.py and has columns:
    dataset, optimizer, seed, eval, best_so_far

The plot shows, for each optimizer, the mean best-so-far CV accuracy as a
function of the number of evaluations, averaged over all datasets and seeds.

Usage:
    python plot_anytime.py                      # reads ./anytime_curves.csv
    python plot_anytime.py path/to/curves.csv   # or a custom path

Output: anytime_performance.png  (Times New Roman, 200 dpi)
=============================================================================
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- config ----
CSV = sys.argv[1] if len(sys.argv) > 1 else "anytime_curves.csv"
OUT = "anytime_performance.png"

# Display order and labels (matches the paper)
ORDER = ["grid", "random", "bo_tpe", "pso", "de", "ga", "sa"]
LABEL = {"grid": "Grid", "random": "Random", "bo_tpe": "BO-TPE",
         "pso": "PSO", "de": "DE", "ga": "GA", "sa": "SA"}
# Distinct colors; grid emphasized, others muted to show it lags
COLOR = {"grid": "#C44E52", "random": "#4C72B0", "bo_tpe": "#55A868",
         "pso": "#8172B3", "de": "#CCB974", "ga": "#64B5CD", "sa": "#937860"}

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

df = pd.read_csv(CSV)

# Average best_so_far across datasets and seeds, per optimizer per evaluation index.
# Mean over (dataset, seed) gives the expected anytime curve.
agg = (df.groupby(["optimizer", "eval"])["best_so_far"]
         .mean()
         .reset_index())

fig, ax = plt.subplots(figsize=(7.0, 4.4))
for opt in ORDER:
    sub = agg[agg.optimizer == opt].sort_values("eval")
    if sub.empty:
        continue
    lw = 2.4 if opt == "grid" else 1.6
    ax.plot(sub["eval"], sub["best_so_far"],
            label=LABEL[opt], color=COLOR.get(opt), linewidth=lw)

ax.set_xlabel("Number of evaluations")
ax.set_ylabel("Mean best-so-far CV accuracy")
# ax.set_title("Anytime performance (mean over datasets and seeds)")
ax.legend(loc="lower right", frameon=False, ncol=2, fontsize=9)
ax.grid(True, alpha=0.25)
# Optional: zoom y-axis to where the action is; comment out if you prefer full range
# ax.set_xlim(1, df["eval"].max())

plt.tight_layout()
plt.savefig(OUT, dpi=200)
print(f"Saved {OUT}")
print(f"  optimizers found: {sorted(df.optimizer.unique())}")
print(f"  max evaluations  : {df['eval'].max()}")
