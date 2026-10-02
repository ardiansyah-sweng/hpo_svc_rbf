"""
apply_metric_patch.py -- make the evaluation metric configurable, so the main
conclusions can be repeated under balanced accuracy (Reviewer 2 comment 6).

Why this metric
---------------
Accuracy is satisfiable by a constant predictor at the majority-class rate. On
ozone-8hr that rate is 0.937 and the best attainable value is 0.947, so the
entire benefit of tuning is under one percentage point; on ilpd it is 1.5
points. Balanced accuracy has a no-skill floor of 1/k for k classes regardless
of class proportions, so it restores headroom on exactly those datasets and
tests whether the conclusions survive a metric that cannot be satisfied by
guessing the majority class.

Two call sites change, both switched by one CONFIG key:

  * the cross-validation objective, via the scoring string
  * the held-out test evaluation, via the scorer function

Default behaviour is UNCHANGED: CONFIG["metric"] is "accuracy", so every
existing run reproduces exactly. The companion runner sets it.

    python apply_metric_patch.py --check
    python apply_metric_patch.py
"""
import argparse
import ast
import shutil
import sys

MARKER = '"metric"'

EDITS = [
    (
        "1. import the balanced-accuracy scorer",
        "from sklearn.metrics import accuracy_score",
        "from sklearn.metrics import accuracy_score, balanced_accuracy_score",
    ),
    (
        "2. cross-validation objective honours the configured metric",
        '        score = cross_val_score(pipe, Xe, ye, cv=self.cv,\n'
        '                                scoring="accuracy",\n',
        '        score = cross_val_score(pipe, Xe, ye, cv=self.cv,\n'
        '                                scoring=self.cfg.get("metric", "accuracy"),\n',
    ),
    (
        "3. held-out test evaluation honours the configured metric",
        "    pipe.fit(X_tr, y_tr)\n"
        "    return accuracy_score(y_te, pipe.predict(X_te))",
        "    pipe.fit(X_tr, y_tr)\n"
        "    pred = pipe.predict(X_te)\n"
        "    if cfg.get(\"metric\", \"accuracy\") == \"balanced_accuracy\":\n"
        "        return balanced_accuracy_score(y_te, pred)\n"
        "    return accuracy_score(y_te, pred)",
    ),
    (
        "4. CONFIG: add the metric key, default unchanged",
        '    "decomp_mode": "baseline",\n',
        '    "decomp_mode": "baseline",\n'
        '    # Reviewer 2 comment 6: "accuracy" reproduces every existing run;\n'
        '    # "balanced_accuracy" repeats the study under a metric whose no-skill\n'
        '    # floor does not depend on the class proportions.\n'
        '    "metric": "accuracy",\n',
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="hpo_benchmark_v3.py")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    src = open(args.file, encoding="utf-8").read()
    if MARKER in src and not args.check:
        print(f"[skip] {args.file} already contains {MARKER}; not patching twice.")
        return

    problems, new = [], src
    for label, old, rep in EDITS:
        n = new.count(old)
        status = "found" if n == 1 else ("MISSING" if n == 0 else f"AMBIGUOUS x{n}")
        print(f"  [{status:>11}] {label}")
        if n != 1:
            problems.append(label)
        else:
            new = new.replace(old, rep, 1)

    if problems:
        print(f"\n{len(problems)} fragment(s) not matched exactly once. Nothing written.")
        sys.exit(1)
    try:
        ast.parse(new)
    except SyntaxError as e:
        print(f"\nPatched source does not parse ({e}). Nothing written.")
        sys.exit(1)
    if args.check:
        print("\nAll fragments found; the patched file would parse. (--check: nothing written)")
        return
    shutil.copyfile(args.file, args.file + ".bak2")
    open(args.file, "w", encoding="utf-8").write(new)
    print(f"\nPatched {args.file}  (backup: {args.file}.bak2)")


if __name__ == "__main__":
    main()