"""
gap_target_analysis.py -- efficiency targets defined on the OPTIMIZATION GAP
rather than as a fraction of accuracy (Reviewer 1 comment 1, Reviewer 2
comment 1).

The problem with the current endpoint
-------------------------------------
Accuracy has a non-zero floor: a constant predictor already scores the majority
class rate b_d. A target of alpha * r_d therefore does not ask an optimizer to
close alpha of the achievable gap, and on a skewed dataset it can sit BELOW b_d,
in which case any configuration at all "reaches" it.

Gap-based target
----------------
    tau_d(alpha) = b_d + alpha * (r_d - b_d)

with b_d the majority-class rate of dataset d and r_d the same robust reference
used before (the best of the per-optimizer median final values). At alpha = 0
the target is the no-skill floor; at alpha = 1 it is the best attainable value.
This is the scale-invariant regret measure Reviewer 2 asks for: reaching
tau_d(alpha) is exactly normalized regret <= 1 - alpha.

Output
------
gap_target_by_run.csv has the same columns as common_target_by_run.csv, so it
can be fed straight to analyze_efficiency_datasetlevel.py:

    python gap_target_analysis.py --curves results/baseline_gridrandom_fullfid/anytime_curves.csv --out results/efficiency_gap
    python analyze_efficiency_datasetlevel.py --by_run results/efficiency_gap/gap_target_by_run.csv --out results/efficiency_gap --target 0.90

Loading the datasets is only needed for the class distribution; no model is fitted.
"""
import argparse
import os

import numpy as np
import pandas as pd

import hpo_benchmark_v3 as hb

ap = argparse.ArgumentParser()
ap.add_argument("--curves", required=True)
ap.add_argument("--out", default=os.path.join("results", "efficiency_gap"))
ap.add_argument("--targets", default="0.90,0.95,0.98")
ap.add_argument("--floor", choices=["majority", "uniform"], default="majority",
                help="no-skill floor b_d: 'majority' = majority-class rate "
                     "(for accuracy), 'uniform' = 1/k (for balanced accuracy, "
                     "whose floor does not depend on the class proportions)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
targets = [float(t) for t in a.targets.split(",")]

df = pd.read_csv(a.curves)
seedcol = "seed" if "seed" in df.columns else "rep"
budget = int(df["eval"].max())

# ---- robust reference r_d (unchanged definition) ---------------------------
final = df[df["eval"] == budget]
per_opt_median = final.groupby(["dataset", "optimizer"])["best_so_far"].median()
reference = per_opt_median.groupby("dataset").max().to_dict()

# ---- no-skill floor b_d ------------------------------
print("Loading datasets for their class distribution (no model is fitted).")
baseline = {}
for entry in hb.CONFIG["datasets"]:
    oid, name = (entry[0], entry[1]) if isinstance(entry, (tuple, list)) else (entry["id"], entry["name"])
    if name not in reference:
        continue
    _, y = hb.load_dataset(oid)
    _, counts = np.unique(y, return_counts=True)
    # A constant predictor scores the majority-class rate under accuracy, but
    # only 1/k under balanced accuracy, which averages recall over the k classes
    # and is therefore blind to the class proportions.
    baseline[name] = (counts.max() / counts.sum() if a.floor == "majority"
                      else 1.0 / len(counts))

missing = sorted(set(reference) - set(baseline))
if missing:
    raise SystemExit(f"no class distribution for {missing}; check CONFIG['datasets'] names")

# ---- diagnostic: was the old target below the no-skill floor? ---------------
rows = []
for ds in sorted(reference):
    b, r = baseline[ds], reference[ds]
    row = dict(dataset=ds, floor=b, floor_rule=a.floor, reference=r, headroom=r - b)
    for al in targets:
        row[f"old_bar_{int(al*100)}"] = al * r
        row[f"gap_bar_{int(al*100)}"] = b + al * (r - b)
        row[f"old_below_floor_{int(al*100)}"] = al * r <= b
    rows.append(row)
diag = pd.DataFrame(rows)
diag.to_csv(os.path.join(a.out, "dataset_baseline.csv"), index=False)

print("\n=== No-skill floor vs the two target definitions ===")
lab = 'major.' if a.floor == 'majority' else '1/k'
print(f"{'dataset':14}{lab:>8}{'ref':>8}{'headroom':>10}", end="")
for al in targets:
    print(f"{'old '+str(int(al*100))+'%':>10}{'gap '+str(int(al*100))+'%':>10}", end="")
print()
for _, r_ in diag.iterrows():
    print(f"{r_.dataset:14}{r_.floor:8.4f}{r_.reference:8.4f}{r_.headroom:10.4f}", end="")
    for al in targets:
        k = int(al * 100)
        flag = "*" if r_[f"old_below_floor_{k}"] else " "
        print(f"{r_['old_bar_'+str(k)]:9.4f}{flag}{r_['gap_bar_'+str(k)]:10.4f}", end="")
    print()
n_triv = sum(diag[f"old_below_floor_{int(al*100)}"].sum() for al in targets)
print(f"\n  * = the fraction-of-accuracy target sits at or below the majority-class rate,")
print(f"      so a constant predictor already attains it. {n_triv} such cases "
      f"across {len(diag)} datasets x {len(targets)} targets.")

# ---- hitting times against the gap-based target ----------------------------
out = []
for (ds, opt, sd), g in df.sort_values("eval").groupby(["dataset", "optimizer", seedcol]):
    b, r = baseline[ds], reference[ds]
    curve, evals = g["best_so_far"].values, g["eval"].values
    for al in targets:
        bar = b + al * (r - b)
        hit = np.argmax(curve >= bar) if np.any(curve >= bar) else None
        out.append(dict(dataset=ds, optimizer=opt, seed=sd, target=al,
                        reference=r, floor=b, bar=bar,
                        evals_to_target=(budget + 1) if hit is None else int(evals[hit]),
                        reached=hit is not None))
by_run = pd.DataFrame(out)
by_run.to_csv(os.path.join(a.out, "gap_target_by_run.csv"), index=False)

summ = []
for (opt, al), g in by_run.groupby(["optimizer", "target"]):
    hit = g.loc[g["reached"], "evals_to_target"]
    summ.append(dict(optimizer=opt, target=al,
                     median_all=g["evals_to_target"].median(),
                     median_reached=hit.median() if len(hit) else np.nan,
                     mean_evals=g["evals_to_target"].mean(),
                     reach_rate=g["reached"].mean(), n_reached=int(g["reached"].sum())))
summary = pd.DataFrame(summ)
summary.to_csv(os.path.join(a.out, "gap_target_summary.csv"), index=False)

for al in targets:
    sub = summary[np.isclose(summary.target, al)].sort_values("median_all")
    print(f"\n=== Gap target = {al:.0%} of the attainable gap ===")
    print(f"  {'optimizer':10}{'median_all':>11}{'median_hit':>11}{'mean':>9}{'reach':>8}")
    for _, r_ in sub.iterrows():
        print(f"  {r_.optimizer:10}{r_.median_all:>11.1f}{r_.median_reached:>11.1f}"
              f"{r_.mean_evals:>9.1f}{r_.reach_rate:>8.0%}")
    if sub.reach_rate.min() < 0.95:
        print(f"  NOTE: reach rate falls to {sub.reach_rate.min():.0%}; median_all is "
              f"inflated by the budget+1 cap. Read median_hit with reach.")

print(f"\nWrote gap_target_by_run.csv, gap_target_summary.csv and dataset_baseline.csv to {a.out}")
print("Next: analyze_efficiency_datasetlevel.py --by_run "
      f"{os.path.join(a.out, 'gap_target_by_run.csv')} --out {a.out} --target 0.90 (and 0.95, 0.98)")