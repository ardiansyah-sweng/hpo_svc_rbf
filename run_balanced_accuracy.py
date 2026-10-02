"""
run_balanced_accuracy.py -- repeat the benchmark under balanced accuracy
(Reviewer 2 comment 6), baseline seed configuration only.

Scope
-----
All nine optimizers, all ten datasets, thirty repetitions: 2,700 runs. Only the
baseline configuration is needed, because the question is whether the main
conclusions hold under a different metric, not how the variance re-partitions
under one.

Grid traversal is randomized in this run. Final solution quality is invariant to
traversal order, so the accuracy-side conclusions are unaffected by that choice,
while the anytime curves become directly usable for the efficiency comparison as
well. One pass therefore serves both axes.

Full-fidelity verification is on, matching the primary analysis.

Output goes to results/balanced_accuracy/, with its own checkpoint directory, so
nothing existing is touched and the resume logic does not skip anything.

    python run_balanced_accuracy.py
    python run_balanced_accuracy.py --out results/balanced_accuracy
"""
import argparse
import copy
import os
import sys

import hpo_benchmark_v3 as hb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("results", "balanced_accuracy"))
    args = ap.parse_args()

    missing = [k for k in ("metric", "full_fidelity_only") if k not in hb.CONFIG]
    if missing:
        sys.exit(f"hpo_benchmark_v3.py is missing {missing}. Run the patches first:\n"
                 "  python apply_sh_fullfid_patch.py\n"
                 "  python apply_metric_patch.py")

    cfg = copy.deepcopy(hb.CONFIG)
    cfg["decomp_mode"] = "baseline"
    cfg["metric"] = "balanced_accuracy"
    cfg["full_fidelity_only"] = True
    cfg["grid_order"] = "random"
    cfg["results_dir"] = args.out
    cfg["checkpoint_dir"] = os.path.join(args.out, "checkpoints")

    print("######## Balanced accuracy, baseline configuration ########")
    print(f"  metric              : {cfg['metric']}")
    print(f"  grid traversal      : {cfg['grid_order']}")
    print(f"  full-fidelity only  : {cfg['full_fidelity_only']}")
    print(f"  output              : {cfg['results_dir']}")
    # hb.OPTIMIZERS lists only seven; cuckoo search and successive halving are
    # passed explicitly, as they are in the main benchmark. Re-running this
    # script against an existing folder recomputes only what the checkpoints
    # do not already contain.
    hb.run_all(cfg, optimizers=["grid", "random", "bo_tpe", "pso", "de", "ga",
                                "sa", "cuckoo", "sh"])

    print("\nDone. The columns are still named test_acc and cv_best; they now hold")
    print("balanced accuracy. Next, repeat the primary analyses on this folder:")
    print(f"  python accuracy_summary.py --summary {os.path.join(args.out, 'summary.csv')} "
          f"--out results/accuracy_balanced")
    print(f"  python analyze_equivalence_tost.py --summary {os.path.join(args.out, 'summary.csv')} "
          f"--out results/equivalence_balanced")


if __name__ == "__main__":
    main()