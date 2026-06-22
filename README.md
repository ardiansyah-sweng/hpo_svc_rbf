# HPO Benchmark for SVC — Grid / Random / BO-TPE / PSO / DE / GA

Reproducible benchmark of hyperparameter optimization algorithms for
Support Vector Classification (RBF kernel) over the (C, gamma) space.

**Paper angle:** cost-aware / anytime performance + dataset-characteristic
algorithm selection, with rigorous statistics (Friedman + Nemenyi + CD diagram).

## Install
```
pip install -r requirements.txt
```

## Run (this is the whole study)
```
python hpo_benchmark.py     # runs all optimizers x datasets x seeds -> results/*.csv
python analyze.py           # stats + figures -> results/ and results/figures/
```

## What you get
- `results/summary.csv` — per run: cv_best, test_acc, best (C,gamma), time, evals_to_99pct
- `results/anytime_curves.csv` — best-so-far vs evaluation (for anytime plot)
- `results/dataset_meta.csv` — n, d, classes, n/d ratio per dataset
- `results/table_test_acc.csv` — mean accuracy (dataset x optimizer)
- `results/nemenyi_test_acc.csv` — post-hoc p-values
- `results/win_table.csv` — winning optimizer per dataset + meta-features
- `results/cost_summary.csv` — mean time & convergence speed
- `results/figures/cd_test_acc.png` — Critical Difference diagram
- `results/figures/anytime_performance.png` — anytime curves

## Fairness guarantees (state these in the paper)
1. Identical search box in log2 space for every optimizer.
2. Identical evaluation budget (CONFIG["budget"] = 100) for every optimizer.
3. Same CV protocol & fixed CV seed during optimization (noise control).
4. 30 independent optimization seeds per (optimizer, dataset) for statistics.

## Configuration
Edit only the `CONFIG` dict at the top of `hpo_benchmark.py`
(search bounds, budget, folds, seeds, dataset list).

## Compute note
Designed for a normal laptop (CPU). Full run (10 datasets x 30 seeds x 6
optimizers x 100 evals) is a few hours — run it overnight. To pilot first,
lower n_seeds to 5 and use 3 datasets.