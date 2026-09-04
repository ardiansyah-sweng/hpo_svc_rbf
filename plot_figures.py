"""
plot_figures.py  --  Generate the two manuscript figures from the experiment output.

Produces:
  fig1_variance.png   Composition of run-to-run variance by dataset (stacked bars).
                      Reads results/variance/variance_components.csv, written by
                      analyze_variance.py.

  fig2_traversal.png  Effect of grid traversal order: efficiency changes by more than
                      an order of magnitude while final accuracy does not.
                      Reads results/grid_order/grid_order_by_run.csv, written by
                      run_grid_order_experiment.py.

USAGE
-----
    python plot_figures.py
    # or with explicit paths / output directory:
    python plot_figures.py --variance results/variance/variance_components.csv \
                           --grid_order results/grid_order/grid_order_by_run.csv \
                           --out figures

Both figures are written at 300 dpi and sized for a single text column of the IJIES
two-column layout (about 3.2 inches wide when placed in the manuscript).
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Colour scheme: print-safe, distinguishable in greyscale.
C_SPLIT = "#2c5f8a"
C_CV = "#8ab4d8"
C_OPT = "#d9d9d9"
BARS = ["#2c5f8a", "#5a8db8", "#8ab4d8", "#c4d9ec"]


def figure_variance(path, out_dir):
    """Stacked bars: share of run-to-run variance per source, per dataset."""
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found; run analyze_variance.py first.")
        return None

    df = pd.read_csv(path)
    need = ["dataset", "share_split_S", "share_cv_F", "share_opt_O"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] {path} is missing columns: {missing}")

    # median share across optimizers, per dataset; sorted by split share
    g = df.groupby("dataset")[["share_split_S", "share_cv_F", "share_opt_O"]].median()
    g = g.sort_values("share_split_S", ascending=False) * 100.0

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(g))
    ax.bar(x, g["share_split_S"], label="Train/test partition",
           color=C_SPLIT, edgecolor="black", linewidth=0.4)
    ax.bar(x, g["share_cv_F"], bottom=g["share_split_S"], label="Cross-validation folds",
           color=C_CV, edgecolor="black", linewidth=0.4)
    ax.bar(x, g["share_opt_O"], bottom=g["share_split_S"] + g["share_cv_F"],
           label="Optimizer initialisation", color=C_OPT, edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(g.index, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Share of run-to-run variance (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.18), frameon=False)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    out = os.path.join(out_dir, "fig1_variance.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    med = df[["share_split_S", "share_cv_F", "share_opt_O"]].median() * 100
    print(f"[OK] {out}")
    print(f"     median shares: split {med['share_split_S']:.1f}%, "
          f"CV {med['share_cv_F']:.1f}%, optimizer {med['share_opt_O']:.1f}%")
    print(f"     split share ranges {g['share_split_S'].min():.1f}% "
          f"({g['share_split_S'].idxmin()}) to {g['share_split_S'].max():.1f}% "
          f"({g['share_split_S'].idxmax()})")
    return out


def figure_traversal(path, out_dir):
    """Two panels: (a) efficiency by traversal policy, (b) accuracy by policy."""
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found; run run_grid_order_experiment.py first.")
        return None

    df = pd.read_csv(path)
    need = ["policy", "evals_to_99pct", "test_acc"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] {path} is missing columns: {missing}")

    order = ["row_major", "col_major", "random", "spiral"]
    labels = ["Row-major", "Column-major", "Random", "Spiral"]
    present = [p for p in order if p in set(df["policy"])]
    if len(present) != len(order):
        print(f"[WARN] policies present: {present}")
        order = present
        labels = [l for p, l in zip(
            ["row_major", "col_major", "random", "spiral"],
            ["Row-major", "Column-major", "Random", "Spiral"]) if p in present]

    med = [df.loc[df.policy == p, "evals_to_99pct"].median() for p in order]
    acc = [df.loc[df.policy == p, "test_acc"].mean() for p in order]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    axes[0].bar(labels, med, color=BARS[:len(order)], edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Median evaluations to target", fontsize=10)
    axes[0].set_title("(a) Search efficiency", fontsize=10)
    ymax = max(med) * 1.20
    for i, v in enumerate(med):
        axes[0].text(i, v + ymax * 0.03, f"{v:.1f}", ha="center", fontsize=9)
    axes[0].set_ylim(0, ymax)
    axes[0].tick_params(labelsize=9, axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[0].set_axisbelow(True)

    axes[1].bar(labels, acc, color=BARS[:len(order)], edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Mean test accuracy", fontsize=10)
    axes[1].set_title("(b) Solution quality", fontsize=10)
    lo, hi = min(acc), max(acc)
    pad = max(0.010, (hi - lo) * 6)
    axes[1].set_ylim(lo - pad, hi + pad)
    for i, v in enumerate(acc):
        axes[1].text(i, v + pad * 0.10, f"{v:.4f}", ha="center", fontsize=8.5)
    axes[1].tick_params(labelsize=9, axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[1].set_axisbelow(True)

    out = os.path.join(out_dir, "fig2_traversal.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] {out}")
    for l, m, a in zip(labels, med, acc):
        print(f"     {l:14} median evals {m:6.1f}   mean accuracy {a:.4f}")
    print(f"     accuracy spread across policies: {max(acc) - min(acc):.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variance", default="results/variance/variance_components.csv")
    ap.add_argument("--grid_order", default="results/grid_order/grid_order_by_run.csv")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("Generating manuscript figures\n")
    figure_variance(args.variance, args.out)
    print()
    figure_traversal(args.grid_order, args.out)
    print("\nInsert fig1_variance.png as Figure 1 (Section 4.2) and "
          "fig2_traversal.png as Figure 2 (Section 4.3).")


if __name__ == "__main__":
    main()