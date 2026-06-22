"""
=============================================================================
Analysis for the HPO-SVC benchmark  (v2 -- two-track)
Run AFTER hpo_benchmark.py has produced results/summary.csv etc.

DESIGN PHILOSOPHY
-----------------
The pilot showed that final-accuracy differences between optimizers are
SMALLER than seed noise, while cost/anytime differences are large and
consistent. This script therefore reports two separate tracks:

  TRACK A -- SOLUTION QUALITY (accuracy): DESCRIPTIVE only.
      Means, per-dataset winners, and an explicit signal-vs-noise table
      (between-optimizer range vs within-optimizer seed SD). We deliberately
      do NOT headline accuracy p-values, because a non-significant result
      here is itself the finding ("optimizer choice barely affects accuracy").

  TRACK B -- COST / EFFICIENCY (anytime, evals-to-target, time): INFERENTIAL.
      Friedman + Nemenyi + CD diagram on cost metrics, where real and
      reportable differences live. This is the paper's main quantitative claim.

Maps directly to paper sections:
  Track A  -> "Optimizer choice does not change final accuracy" (RQ1)
  Track B  -> "Optimizer choice strongly affects efficiency"     (RQ2)
  win_table-> "Which optimizer for which dataset"                (RQ3 / selection)
=============================================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
import scikit_posthocs as sp

RESULTS = "results"
FIGS = os.path.join(RESULTS, "figures")
os.makedirs(FIGS, exist_ok=True)

OPT_ORDER = ["grid", "random", "bo_tpe", "pso", "de", "ga", "sa"]
OPT_LABEL = {"grid": "Grid", "random": "Random", "bo_tpe": "BO-TPE",
             "pso": "PSO", "de": "DE", "ga": "GA", "sa": "SA"}

# Significance threshold for the inferential (cost) track
ALPHA = 0.05


# =============================================================================
# IO
# =============================================================================
def load():
    df = pd.read_csv(os.path.join(RESULTS, "summary.csv"))
    dfc = pd.read_csv(os.path.join(RESULTS, "anytime_curves.csv"))
    meta_path = os.path.join(RESULTS, "dataset_meta.csv")
    dfm = pd.read_csv(meta_path) if os.path.exists(meta_path) else None
    return df, dfc, dfm


def _labeled(piv):
    return piv[OPT_ORDER].rename(columns=OPT_LABEL)


# =============================================================================
# TRACK A -- SOLUTION QUALITY (descriptive)
# =============================================================================
def accuracy_descriptive(df, metric="test_acc"):
    print("\n" + "=" * 70)
    print(f"TRACK A  |  SOLUTION QUALITY  ({metric})  --  DESCRIPTIVE")
    print("=" * 70)

    # mean +/- sd per optimizer (pooled over all dataset,seed)
    g = df.groupby("optimizer")[metric].agg(["mean", "std"])
    g = g.reindex(OPT_ORDER).rename(index=OPT_LABEL)
    print("\nPooled mean +/- SD per optimizer:")
    print(g.round(4))

    # per-dataset mean table
    piv = _labeled(df.groupby(["dataset", "optimizer"])[metric].mean().unstack())
    piv.to_csv(os.path.join(RESULTS, f"A_table_{metric}.csv"))
    print(f"\nMean {metric} (dataset x optimizer):")
    print(piv.round(4))

    # per-dataset winner (report, but with caveat from noise table below)
    winners = piv.idxmax(axis=1)
    print("\nNominal winner per dataset (treat with caution -- see noise table):")
    print(winners.to_string())

    return piv, g


def signal_vs_noise(df, metric="test_acc"):
    """The key honesty table: is between-optimizer spread bigger than seed noise?"""
    print("\n--- Signal-vs-Noise (per dataset) ---")
    rows = []
    for d in sorted(df.dataset.unique()):
        sub = df[df.dataset == d]
        opt_means = sub.groupby("optimizer")[metric].mean()
        between = float(opt_means.max() - opt_means.min())
        within = float(sub.groupby("optimizer")[metric].std().mean())
        ratio = between / within if within > 0 else np.nan
        rows.append({"dataset": d, "between_opt_range": round(between, 4),
                     "within_opt_seed_SD": round(within, 4),
                     "signal_noise_ratio": round(ratio, 3)})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(RESULTS, f"A_signal_vs_noise_{metric}.csv"), index=False)
    print(tbl.to_string(index=False))
    verdict = ("Between-optimizer differences are SMALLER than seed noise "
               "in most datasets -> accuracy differences are not reliable."
               if (tbl.signal_noise_ratio < 1).mean() >= 0.5 else
               "Between-optimizer differences exceed seed noise in most "
               "datasets -> accuracy differences may be meaningful.")
    print("\nVerdict:", verdict)
    return tbl


def accuracy_optional_friedman(df, metric="test_acc"):
    """Reported only as a footnote; non-significance is expected & is the point."""
    piv = df.groupby(["dataset", "optimizer"])[metric].mean().unstack()[OPT_ORDER]
    n_blocks = piv.shape[0]
    if n_blocks < 3:
        print(f"\n(Friedman skipped for {metric}: need >=3 datasets.)")
        return
    stat, p = friedmanchisquare(*[piv[o].values for o in OPT_ORDER])
    print(f"\n(Footnote) Friedman on accuracy across {n_blocks} datasets: "
          f"chi2={stat:.3f}, p={p:.3f} "
          f"{'-- NOT significant (expected)' if p >= ALPHA else '-- significant'}")
    if n_blocks < 8:
        print("  NOTE: with <8 datasets this test is underpowered; do not "
              "headline this result.")


# =============================================================================
# TRACK B -- COST / EFFICIENCY (inferential)
# =============================================================================
def cost_inferential(df, metric, lower_is_better=True):
    """Friedman + Nemenyi on a cost metric (e.g. evals_to_99pct, time_sec).
    Uses (dataset x seed) blocks for power."""
    print("\n" + "=" * 70)
    print(f"TRACK B  |  COST / EFFICIENCY  ({metric})  --  INFERENTIAL")
    print("=" * 70)

    piv = df.pivot_table(index=["dataset", "seed"], columns="optimizer",
                         values=metric)[OPT_ORDER].dropna()
    n_blocks = piv.shape[0]
    print(f"\nBlocks (dataset x seed): {n_blocks}")

    # pooled means
    g = df.groupby("optimizer")[metric].agg(["mean", "std"]).reindex(OPT_ORDER)
    g = g.rename(index=OPT_LABEL)
    print(f"\nPooled {metric} per optimizer:")
    print(g.round(3))

    # Friedman
    stat, p = friedmanchisquare(*[piv[o].values for o in OPT_ORDER])
    print(f"\nFriedman: chi2={stat:.3f}, p={p:.4g} "
          f"{'-- SIGNIFICANT' if p < ALPHA else '-- not significant'}")

    # average ranks (sign depends on direction)
    sign = 1.0 if lower_is_better else -1.0
    R = np.array([rankdata(sign * piv.iloc[i].values) for i in range(n_blocks)])
    ranks = pd.Series(R.mean(0), index=[OPT_LABEL[o] for o in OPT_ORDER]).sort_values()
    print("\nAverage ranks (lower rank = better):")
    print(ranks.round(3).to_string())

    nem = None
    if p < ALPHA:
        nem = sp.posthoc_nemenyi_friedman(piv.values)
        nem.index = [OPT_LABEL[o] for o in OPT_ORDER]
        nem.columns = [OPT_LABEL[o] for o in OPT_ORDER]
        nem.to_csv(os.path.join(RESULTS, f"B_nemenyi_{metric}.csv"))
        print("\nNemenyi post-hoc p-values:")
        print(nem.round(4))
        _cd_plot(ranks, metric)
    else:
        print("\n(Skipping Nemenyi/CD: omnibus not significant.)")

    g.to_csv(os.path.join(RESULTS, f"B_cost_{metric}.csv"))
    return ranks, p, nem


def _cd_plot(ranks, metric):
    plt.figure(figsize=(8, 2.2))
    try:
        sp.critical_difference_diagram(ranks, sp.posthoc_nemenyi_friedman)
    except Exception:
        ranks.sort_values(ascending=False).plot.barh()
        plt.xlabel("Average rank")
    plt.title(f"Critical Difference -- {metric}")
    plt.tight_layout()
    out = os.path.join(FIGS, f"B_cd_{metric}.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


# =============================================================================
# ANYTIME CURVES (visual, supports Track B)
# =============================================================================
def anytime_plot(dfc):
    plt.figure(figsize=(7, 4.5))
    for o in OPT_ORDER:
        s = dfc[dfc.optimizer == o].groupby("eval")["best_so_far"].mean()
        plt.plot(s.index, s.values, label=OPT_LABEL[o], linewidth=1.8)
    plt.xlabel("Objective evaluations")
    plt.ylabel("Mean best-so-far CV accuracy")
    plt.title("Anytime performance (mean over datasets & seeds)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIGS, "anytime_performance.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"\nSaved {out}")


def anytime_checkpoints(dfc, points=(5, 10, 20, 50, 100)):
    rows = {}
    for o in OPT_ORDER:
        s = dfc[dfc.optimizer == o].groupby("eval")["best_so_far"].mean()
        rows[OPT_LABEL[o]] = [round(float(s.loc[c]), 4) for c in points if c in s.index]
    tbl = pd.DataFrame(rows, index=[f"eval{c}" for c in points]).T
    tbl.to_csv(os.path.join(RESULTS, "B_anytime_checkpoints.csv"))
    print("\nAnytime best-so-far at checkpoints:")
    print(tbl.to_string())
    return tbl


# =============================================================================
# ALGORITHM SELECTION VIEW (Track C / RQ3)
# =============================================================================
def win_table(df, dfm, metric="test_acc"):
    if dfm is None:
        print("\n(win_table skipped: dataset_meta.csv not found.)")
        return None
    print("\n" + "=" * 70)
    print("ALGORITHM-SELECTION VIEW  (winner vs dataset meta-features)")
    print("=" * 70)
    piv = df.groupby(["dataset", "optimizer"])[metric].mean().unstack()[OPT_ORDER]
    winners = piv.idxmax(axis=1).map(OPT_LABEL)
    tbl = dfm.set_index("dataset").copy()
    tbl["winner_acc"] = winners
    # efficiency winner (fewest evals to 99%)
    eff = df.groupby(["dataset", "optimizer"])["evals_to_99pct"].mean().unstack()[OPT_ORDER]
    tbl["winner_efficiency"] = eff.idxmin(axis=1).map(OPT_LABEL)
    tbl["size_class"] = np.where(tbl["n_samples"] < 1000, "small", "mid+")
    tbl["dim_class"] = np.where(tbl["n_features"] < 20, "low", "high")
    keep = ["n_samples", "n_features", "n_classes", "ratio_nd",
            "size_class", "dim_class", "winner_acc", "winner_efficiency"]
    tbl = tbl[[c for c in keep if c in tbl.columns]]
    tbl.to_csv(os.path.join(RESULTS, "C_win_table.csv"))
    print(tbl.to_string())
    return tbl


# =============================================================================
# MAIN
# =============================================================================
def main():
    df, dfc, dfm = load()

    n_datasets = df.dataset.nunique()
    print(f"\nLoaded {len(df)} runs | {n_datasets} datasets | "
          f"{df.optimizer.nunique()} optimizers | "
          f"{df.groupby(['dataset','optimizer']).size().max()} seeds/cell")
    if n_datasets < 8:
        print("\n*** PILOT MODE WARNING ***")
        print(f"Only {n_datasets} datasets. Inferential tests are underpowered.")
        print("Track A (accuracy) stays descriptive. Track B p-values are")
        print("indicative only until the full (>=10 dataset) run.\n")

    # TRACK A: accuracy, descriptive
    accuracy_descriptive(df, "test_acc")
    signal_vs_noise(df, "test_acc")
    accuracy_optional_friedman(df, "test_acc")

    # TRACK B: cost metrics, inferential
    cost_inferential(df, "evals_to_99pct", lower_is_better=True)
    cost_inferential(df, "time_sec", lower_is_better=True)
    anytime_plot(dfc)
    anytime_checkpoints(dfc)

    # TRACK C: selection view
    win_table(df, dfm, "test_acc")

    print("\n" + "=" * 70)
    print("Done. Outputs in:", RESULTS)
    print("  A_*  = accuracy (descriptive)")
    print("  B_*  = cost/efficiency (inferential)")
    print("  C_*  = algorithm-selection view")
    print("=" * 70)


if __name__ == "__main__":
    main()
