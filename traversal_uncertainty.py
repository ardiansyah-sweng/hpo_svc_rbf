"""
traversal_uncertainty.py -- how much of the grid-search efficiency result comes
from the traversal permutation itself? (Reviewer 1 comment 8)

The order-generation rule
-------------------------
run_grid() builds the lattice row-major and then, for grid_order="random",
permutes it with

    rng = np.random.RandomState(seed); idx = rng.permutation(len(pts))

where `seed` is the seed_opt of the repetition. So the permutation is NOT a
fourth independent stream: it is drawn from the optimizer-initialization stream,
which grid search has no other use for. The reviewer's premise that it is
"explicitly separate from the optimizer-initialization seed" is therefore not
the case, and the three-source framework stays intact.

One scope note follows from this. The variance attribution (Table 7) was run
with grid_order="row_major", where grid is deterministic and its
optimizer-initialization variance is zero. The efficiency analysis uses the
randomized traversal, where that same stream drives the permutation.

Two quantities are computed
---------------------------
1. How much of grid's run-to-run efficiency variance is the permutation?
   Under a deterministic policy, variation across repetitions comes only from
   the train/test split and the CV folds. Under the random policy it also
   includes the permutation. The difference estimates the permutation share:

       share = (Var_random - mean Var_deterministic) / Var_random

   This assumes the two contributions add, the same assumption checked in
   Section 4.2, and is reported as an estimate rather than a decomposition.

2. A bootstrap confidence interval for grid's dataset-level median
   evaluations-to-target, i.e. the number that appears in Table 8. Repetitions
   are resampled within each dataset, so the interval covers the permutation
   together with the split and fold draws.

    python traversal_uncertainty.py --grid_order results/grid_order/grid_order_by_run.csv --by_run results/efficiency_gridrandom_fullfid/common_target_by_run.csv --out results/traversal_uncertainty
"""
import argparse
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--grid_order", required=True)
ap.add_argument("--by_run", required=True)
ap.add_argument("--out", default=os.path.join("results", "traversal_uncertainty"))
ap.add_argument("--targets", default="0.90,0.95,0.98")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(a.seed)

# ---------- 1. permutation share of grid's efficiency variance --------------
g = pd.read_csv(a.grid_order)
DET = ["row_major", "col_major", "spiral"]
rows = []
for ds, d in g.groupby("dataset"):
    v = d.groupby("policy")["evals_to_99pct"].var(ddof=1)
    if "random" not in v.index:
        continue
    v_rand = v["random"]
    v_det = v.reindex([p for p in DET if p in v.index]).mean()
    share = np.nan if v_rand <= 0 else max(0.0, (v_rand - v_det) / v_rand)
    rows.append(dict(dataset=ds, var_random=v_rand, var_deterministic_mean=v_det,
                     sd_random=np.sqrt(v_rand), sd_deterministic_mean=np.sqrt(v_det),
                     permutation_share=share))
share = pd.DataFrame(rows).sort_values("permutation_share", ascending=False)
share.to_csv(os.path.join(a.out, "permutation_share.csv"), index=False)

print("=== 1. Share of grid's run-to-run efficiency variance due to the permutation ===")
print("    (random policy vs the mean of the three deterministic policies)\n")
print(f"{'dataset':14}{'SD random':>11}{'SD determ.':>12}{'perm. share':>13}")
for _, r in share.iterrows():
    s = "   n/a" if np.isnan(r.permutation_share) else f"{r.permutation_share:12.0%}"
    print(f"{r.dataset:14}{r.sd_random:11.1f}{r.sd_deterministic_mean:12.1f}{s}")
med = share["permutation_share"].median()
print(f"\n  median share across datasets: {med:.0%}"
      f"   (datasets where it exceeds half: "
      f"{int((share.permutation_share > 0.5).sum())}/{len(share)})")

# ---------- 2. bootstrap CI for grid's dataset-level median -----------------
b = pd.read_csv(a.by_run)
seedcol = "seed" if "seed" in b.columns else "rep"
targets = [float(t) for t in a.targets.split(",")]
out = []
print("\n=== 2. Bootstrap CI for grid's dataset-level median evals-to-target ===")
print("    (repetitions resampled within each dataset; covers permutation + split + folds)\n")
print(f"{'target':>8}{'point':>9}{'95% CI':>18}{'CI width':>10}")
for t in targets:
    d = b[(b.optimizer == "grid") & (np.isclose(b.target, t))]
    per_ds = {ds: gg["evals_to_target"].values for ds, gg in d.groupby("dataset")}
    point = np.median([np.median(v) for v in per_ds.values()])
    draws = np.empty(a.boot)
    for i in range(a.boot):
        meds = [np.median(rng.choice(v, size=len(v), replace=True)) for v in per_ds.values()]
        draws[i] = np.median(meds)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    out.append(dict(target=t, point=point, ci_low=lo, ci_high=hi, width=hi - lo))
    print(f"{t:>8.2f}{point:>9.1f}{f'[{lo:.1f}, {hi:.1f}]':>18}{hi-lo:>10.1f}")
pd.DataFrame(out).to_csv(os.path.join(a.out, "grid_bootstrap_ci.csv"), index=False)

print(f"\nWrote permutation_share.csv and grid_bootstrap_ci.csv to {a.out}")
print("Read the CI against the adaptive optimizers' medians in Table 8: if they fall")
print("inside it, grid's position is not distinguishable from theirs under resampling.")