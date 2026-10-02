"""
check_env_equivalence.py -- is this Python environment numerically equivalent to
the one that produced results/baseline/summary.csv?

Grid search is deterministic given the lattice, traversal order, split and CV
seeds, so rerunning it here must reproduce the stored values EXACTLY. If it
does, the rerun of SH can be merged with the eight stored optimizers. If it does
not, a library version changed (most likely scikit-learn) and the SH rerun would
not be comparable with the stored results.

Runs grid on one dataset for three repetitions (a few minutes at most) into a
throw-away folder. Nothing in results/baseline is touched.

    python check_env_equivalence.py
    python check_env_equivalence.py --dataset diabetes --reps 5
"""
import argparse
import copy
import os
import shutil
import sys

import numpy as np
import pandas as pd

import hpo_benchmark_v3 as hb

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="wdbc")
p.add_argument("--reps", type=int, default=3)
p.add_argument("--stored", default=os.path.join("results", "baseline", "summary.csv"))
a = p.parse_args()

ds = [d for d in hb.CONFIG["datasets"] if d[1] == a.dataset]
if not ds:
    sys.exit(f"dataset '{a.dataset}' not in CONFIG['datasets']")

tmp = os.path.join("results", "_envcheck")
shutil.rmtree(tmp, ignore_errors=True)
cfg = copy.deepcopy(hb.CONFIG)
cfg.update(decomp_mode="baseline", grid_order="row_major",   # the stored baseline
           results_dir=tmp, checkpoint_dir=os.path.join(tmp, "ckpt"))
new, _, _ = hb.run_all(cfg, datasets=ds, optimizers=["grid"],
                       n_seeds=a.reps, verbose=False)

old = pd.read_csv(a.stored)
old = old[(old.dataset == a.dataset) & (old.optimizer == "grid") & (old.rep < a.reps)]
key = ["rep"]
cols = ["cv_best", "test_acc", "best_log2C", "best_log2gamma"]
m = old[key + cols].merge(new[key + cols], on=key, suffixes=("_stored", "_now"))

print(f"\n=== grid on {a.dataset}, reps 0..{a.reps-1}: stored vs this environment ===")
ok = True
for c in cols:
    d = np.abs(m[c + "_stored"] - m[c + "_now"]).max()
    same = d < 1e-12
    ok &= same
    print(f"  {c:15} max |diff| = {d:.2e}   {'identical' if same else 'DIFFERENT'}")

print("\nVERDICT:", "EQUIVALENT - safe to rerun SH and merge." if ok else
      "NOT EQUIVALENT - a library version differs; do not merge a new SH run "
      "with the stored optimizers. Report the versions below.")
import sklearn
print(f"  python {sys.version.split()[0]} | scikit-learn {sklearn.__version__} "
      f"| numpy {np.__version__} | pandas {pd.__version__}")
shutil.rmtree(tmp, ignore_errors=True)