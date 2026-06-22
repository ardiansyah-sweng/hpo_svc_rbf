"""
=============================================================================
Analysis for the HPO-SVC benchmark.
Run AFTER hpo_benchmark.py has produced results/summary.csv etc.
Generates: Friedman test, Nemenyi post-hoc, Critical Difference diagram,
           anytime-performance plot, and a per-dataset win table linked to
           dataset meta-features (the "algorithm selection" angle).
=============================================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, rankdata
import scikit_posthocs as sp

RESULTS = "results"
FIGS = os.path.join(RESULTS, "figures")
os.makedirs(FIGS, exist_ok=True)

OPT_ORDER = ["grid", "random", "bo_tpe", "pso", "de", "ga"]
OPT_LABEL = {"grid": "Grid", "random": "Random", "bo_tpe": "BO-TPE",
             "pso": "PSO", "de": "DE", "ga": "GA"}


def load():
    df = pd.read_csv(os.path.join(RESULTS, "summary.csv"))
    dfc = pd.read_csv(os.path.join(RESULTS, "anytime_curves.csv"))
    dfm = pd.read_csv(os.path.join(RESULTS, "dataset_meta.csv"))
    return df, dfc, dfm


# -----------------------------------------------------------------------------
# 1. Mean test accuracy table (dataset x optimizer)
# -----------------------------------------------------------------------------
def accuracy_table(df, metric="test_acc"):
    piv = df.groupby(["dataset", "optimizer"])[metric].mean().unstack()
    piv = piv[OPT_ORDER].rename(columns=OPT_LABEL)
    piv.to_csv(os.path.join(RESULTS, f"table_{metric}.csv"))
    print(f"\n=== Mean {metric} (dataset x optimizer) ===")
    print(piv.round(4))
    return piv


# -----------------------------------------------------------------------------
# 2. Friedman + Nemenyi over datasets (using per-dataset mean accuracy)
# -----------------------------------------------------------------------------
def friedman_nemenyi(df, metric="test_acc"):
    piv = df.groupby(["dataset", "optimizer"])[metric].mean().unstack()[OPT_ORDER]
    data = [piv[o].values for o in OPT_ORDER]
    stat, p = friedmanchisquare(*data)
    print(f"\n=== Friedman ({metric}) ===")
    print(f"chi2 = {stat:.4f}, p = {p:.4g}")

    # Average ranks (higher accuracy -> rank 1)
    ranks = np.zeros((piv.shape[0], len(OPT_ORDER)))
    for i in range(piv.shape[0]):
        ranks[i] = rankdata(-piv.iloc[i].values)
    avg_ranks = ranks.mean(axis=0)
    rank_series = pd.Series(avg_ranks, index=[OPT_LABEL[o] for o in OPT_ORDER]
                            ).sort_values()
    print("\nAverage ranks (lower = better):")
    print(rank_series.round(3))

    nem = sp.posthoc_nemenyi_friedman(piv.values)
    nem.index = [OPT_LABEL[o] for o in OPT_ORDER]
    nem.columns = [OPT_LABEL[o] for o in OPT_ORDER]
    nem.to_csv(os.path.join(RESULTS, f"nemenyi_{metric}.csv"))
    print("\nNemenyi post-hoc p-values:")
    print(nem.round(4))
    return rank_series, nem, p


# -----------------------------------------------------------------------------
# 3. Critical Difference diagram
# -----------------------------------------------------------------------------
def cd_diagram(rank_series, metric="test_acc"):
    plt.figure(figsize=(8, 2.2))
    try:
        sp.critical_difference_diagram(rank_series, sp.posthoc_nemenyi_friedman)
    except Exception:
        # fallback: simple rank barh
        rank_series.sort_values().plot.barh()
        plt.xlabel("Average rank")
    plt.title(f"Critical Difference ({metric})")
    plt.tight_layout()
    out = os.path.join(FIGS, f"cd_{metric}.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


# -----------------------------------------------------------------------------
# 4. Anytime performance plot (mean best-so-far vs evals), averaged over data+seed
# -----------------------------------------------------------------------------
def anytime_plot(dfc):
    plt.figure(figsize=(7, 4.5))
    for o in OPT_ORDER:
        sub = dfc[dfc.optimizer == o].groupby("eval")["best_so_far"].mean()
        plt.plot(sub.index, sub.values, label=OPT_LABEL[o], linewidth=1.8)
    plt.xlabel("Objective evaluations")
    plt.ylabel("Mean best-so-far CV accuracy")
    plt.title("Anytime performance (averaged over datasets & seeds)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIGS, "anytime_performance.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


# -----------------------------------------------------------------------------
# 5. Win table linked to dataset meta-features (algorithm-selection angle)
# -----------------------------------------------------------------------------
def win_table(df, dfm, metric="test_acc"):
    piv = df.groupby(["dataset", "optimizer"])[metric].mean().unstack()[OPT_ORDER]
    winners = piv.idxmax(axis=1).map(OPT_LABEL)
    tbl = dfm.set_index("dataset").copy()
    tbl["winner"] = winners
    tbl["best_acc"] = piv.max(axis=1).round(4)
    # simple strata
    tbl["size_class"] = np.where(tbl["n_samples"] < 1000, "small", "mid+")
    tbl["dim_class"] = np.where(tbl["n_features"] < 20, "low", "high")
    cols = ["n_samples", "n_features", "n_classes", "ratio_nd",
            "size_class", "dim_class", "winner", "best_acc"]
    tbl = tbl[cols]
    tbl.to_csv(os.path.join(RESULTS, "win_table.csv"))
    print("\n=== Winner per dataset (algorithm-selection view) ===")
    print(tbl)
    print("\nWinner counts by stratum:")
    print(tbl.groupby(["size_class", "dim_class"])["winner"].value_counts())
    return tbl


# -----------------------------------------------------------------------------
# 6. Cost-aware summary: time + evals-to-99%
# -----------------------------------------------------------------------------
def cost_summary(df):
    g = df.groupby("optimizer")[["time_sec", "evals_to_99pct"]].mean()
    g = g.reindex(OPT_ORDER).rename(index=OPT_LABEL)
    g.to_csv(os.path.join(RESULTS, "cost_summary.csv"))
    print("\n=== Cost-aware summary (mean over all runs) ===")
    print(g.round(3))
    return g


def main():
    df, dfc, dfm = load()
    accuracy_table(df, "test_acc")
    accuracy_table(df, "cv_best")
    ranks, nem, p = friedman_nemenyi(df, "test_acc")
    cd_diagram(ranks, "test_acc")
    anytime_plot(dfc)
    win_table(df, dfm, "test_acc")
    cost_summary(df)
    print("\nAll analysis outputs written to:", RESULTS)


if __name__ == "__main__":
    main()
