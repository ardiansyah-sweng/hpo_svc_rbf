"""
run_sh_fullfid.py -- rerun ONLY Successive Halving with full-fidelity
verification, after apply_sh_fullfid_patch.py has been applied.

The eight full-fidelity optimizers do not need to be rerun: every one of their
evaluations is at rho = 1, so the flag cannot change their results.

Output goes to NEW folders, results/sh_fullfid/<mode>/, with their own
checkpoint directory. This matters: the existing checkpoints already contain SH,
and the resume logic would silently skip it if pointed at the old folders.

Usage
-----
    python run_sh_fullfid.py                      # baseline only (300 runs)
    python run_sh_fullfid.py --modes all          # baseline + 3 isolation modes (1,200 runs)
    python run_sh_fullfid.py --modes baseline,isolate_S

The baseline run is what the accuracy and efficiency analyses need. The three
isolation modes are needed only to refresh SH's contribution to Table 7.
"""
import argparse
import copy
import os
import sys

import hpo_benchmark_v3 as hb

ALL_MODES = ["baseline", "isolate_S", "isolate_F", "isolate_O"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="baseline",
                    help="comma-separated modes, or 'all'")
    ap.add_argument("--out", default=os.path.join("results", "sh_fullfid"))
    args = ap.parse_args()

    if "full_fidelity_only" not in hb.CONFIG:
        sys.exit("hpo_benchmark_v3.py is not patched yet. "
                 "Run: python apply_sh_fullfid_patch.py")

    modes = ALL_MODES if args.modes == "all" else [m.strip() for m in args.modes.split(",")]
    bad = [m for m in modes if m not in ALL_MODES]
    if bad:
        sys.exit(f"unknown mode(s): {bad}; choose from {ALL_MODES}")

    for mode in modes:
        cfg = copy.deepcopy(hb.CONFIG)
        cfg["decomp_mode"] = mode
        cfg["full_fidelity_only"] = True
        cfg["results_dir"] = os.path.join(args.out, mode)
        cfg["checkpoint_dir"] = os.path.join(cfg["results_dir"], "checkpoints")
        print(f"\n######## SH, full-fidelity verification, mode = {mode} ########")
        print(f"         output -> {cfg['results_dir']}")
        hb.run_all(cfg, optimizers=["sh"])

    print("\nDone. Next: python merge_sh_fullfid.py  (see its docstring)")


if __name__ == "__main__":
    main()