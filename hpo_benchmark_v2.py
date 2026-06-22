"""
=============================================================================
HPO Benchmark for SVC  (v2 -- parallel + checkpointed)
Grid / Random / Bayesian (TPE) / PSO / DE / GA  over (C, gamma), RBF kernel.

CHANGES vs v1:
  * CV parallelism via CONFIG["cv_n_jobs"] (use -1 for all cores).
  * Per-DATASET checkpointing: each dataset's results are written to
    results/checkpoints/<dataset>.csv (+ _curves, _meta) as soon as it
    finishes. On restart the script SKIPS datasets already completed.
  * Final merge step rebuilds the same summary.csv / anytime_curves.csv /
    dataset_meta.csv as v1, so analyze_v2.py works unchanged.

WHY CV-level parallelism (not seed-level):
  The Objective records evaluation history in order; parallelizing across
  seeds would require process isolation and complicate the anytime curves.
  Parallelizing the k-fold CV inside each evaluation is a single, safe knob
  that uses all cores during the expensive SVM fits, which dominate runtime.
=============================================================================
"""

import os
import time
import glob
import warnings
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "log2C_bounds":     (-5.0, 15.0),
    "log2gamma_bounds": (-15.0, 3.0),
    "budget": 100,
    "cv_folds": 5,
    "cv_seed": 42,
    "n_seeds": 30,
    "test_size": 0.20,

    # NEW: parallelism for the k-fold CV inside each objective evaluation.
    # -1 = all logical cores. On an i7-8665U (4c/8t) this gives ~3-4x.
    "cv_n_jobs": -1,

    "results_dir": "results",
    "checkpoint_dir": os.path.join("results", "checkpoints"),

    "datasets": [
        (37,   "diabetes"),
        (15,   "breast-w"),
        (54,   "vehicle"),
        (1510, "wdbc"),
        (40975,"car"),
        (1480, "ilpd"),
        (1494, "qsar-biodeg"),
        (1462, "banknote"),
        (40982,"steel-faults"),
        (1487, "ozone-8hr"),
    ],

    # NEW: optional cap on training-set size to bound SVM cost (None = no cap).
    # If set, the TRAIN split is subsampled (stratified) to at most this many
    # rows BEFORE optimization. State this in the paper's methodology.
    "max_train_n": None,
}

OPTIMIZERS = ["grid", "random", "bo_tpe", "pso", "de", "ga", "sa"]


# =============================================================================
# OBJECTIVE
# =============================================================================
class Objective:
    def __init__(self, X, y, cfg):
        self.X = X
        self.y = y
        self.cfg = cfg
        self.cv = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True,
                                  random_state=cfg["cv_seed"])
        self.history = []
        self.n_eval = 0

    def evaluate(self, log2C, log2g):
        if self.n_eval >= self.cfg["budget"]:
            return max((h[3] for h in self.history), default=0.0)
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("sc", StandardScaler()),
            ("svc", SVC(C=2.0 ** log2C, gamma=2.0 ** log2g, kernel="rbf")),
        ])
        score = cross_val_score(pipe, self.X, self.y, cv=self.cv,
                                scoring="accuracy",
                                n_jobs=self.cfg.get("cv_n_jobs", 1)).mean()
        self.n_eval += 1
        self.history.append((self.n_eval, log2C, log2g, score))
        return score

    def best_so_far_curve(self):
        curve = np.full(self.cfg["budget"], np.nan)
        best = -np.inf
        for (idx, _, _, s) in self.history:
            best = max(best, s)
            curve[idx - 1] = best
        last = -np.inf
        for i in range(len(curve)):
            if np.isnan(curve[i]):
                curve[i] = last
            else:
                last = curve[i]
        return curve

    def best_config(self):
        if not self.history:
            return None
        return max(self.history, key=lambda h: h[3])


# =============================================================================
# OPTIMIZERS
# =============================================================================
def run_grid(obj, cfg, seed):
    lc = cfg["log2C_bounds"]; lg = cfg["log2gamma_bounds"]
    side = int(round(cfg["budget"] ** 0.5))
    for c in np.linspace(lc[0], lc[1], side):
        for g in np.linspace(lg[0], lg[1], side):
            if obj.n_eval >= cfg["budget"]:
                return
            obj.evaluate(c, g)


def run_random(obj, cfg, seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    lc = cfg["log2C_bounds"]; lg = cfg["log2gamma_bounds"]
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.RandomSampler(seed=seed))
    study.optimize(lambda t: obj.evaluate(
        t.suggest_float("log2C", lc[0], lc[1]),
        t.suggest_float("log2g", lg[0], lg[1])),
        n_trials=cfg["budget"], show_progress_bar=False)


def run_bo_tpe(obj, cfg, seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    lc = cfg["log2C_bounds"]; lg = cfg["log2gamma_bounds"]
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda t: obj.evaluate(
        t.suggest_float("log2C", lc[0], lc[1]),
        t.suggest_float("log2g", lg[0], lg[1])),
        n_trials=cfg["budget"], show_progress_bar=False)


def _pymoo_problem(obj, cfg):
    from pymoo.core.problem import ElementwiseProblem
    lc = cfg["log2C_bounds"]; lg = cfg["log2gamma_bounds"]

    class SVCProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=2, n_obj=1, n_constr=0,
                             xl=np.array([lc[0], lg[0]]),
                             xu=np.array([lc[1], lg[1]]))

        def _evaluate(self, x, out, *a, **k):
            out["F"] = -obj.evaluate(x[0], x[1])
    return SVCProblem()


def _run_pymoo(obj, cfg, seed, algorithm):
    from pymoo.optimize import minimize
    minimize(_pymoo_problem(obj, cfg), algorithm,
             termination=("n_evals", cfg["budget"]), seed=seed, verbose=False)


def run_pso(obj, cfg, seed):
    from pymoo.algorithms.soo.nonconvex.pso import PSO
    _run_pymoo(obj, cfg, seed, PSO(pop_size=20))


def run_de(obj, cfg, seed):
    from pymoo.algorithms.soo.nonconvex.de import DE
    _run_pymoo(obj, cfg, seed, DE(pop_size=20))


def run_ga(obj, cfg, seed):
    from pymoo.algorithms.soo.nonconvex.ga import GA
    _run_pymoo(obj, cfg, seed, GA(pop_size=20))


def run_sa(obj, cfg, seed):
    """Simulated Annealing for the continuous (log2C, log2gamma) space.

    A classic trajectory-based metaheuristic (single solution), in contrast to
    the population-based PSO/DE/GA. Fairness: identical budget (cfg['budget']
    objective evaluations) and identical search box. Design choices kept
    standard and un-tuned to avoid advantaging SA:
      * Gaussian neighborhood with step = 10% of each dimension's range.
      * Geometric cooling T_k = T0 * alpha^k, with T0 and alpha set so that
        the acceptance probability starts permissive and ends near-greedy
        across exactly `budget` evaluations.
      * Maximization: accept worse moves with prob exp(delta / T) where
        delta = f_new - f_cur (negative for worse).
    """
    rng = np.random.RandomState(seed)
    lc = cfg["log2C_bounds"]; lg = cfg["log2gamma_bounds"]
    budget = cfg["budget"]
    span = np.array([lc[1] - lc[0], lg[1] - lg[0]], dtype=float)
    lo = np.array([lc[0], lg[0]], dtype=float)
    hi = np.array([lc[1], lg[1]], dtype=float)
    step = 0.10 * span                       # neighborhood scale

    # cooling schedule over `budget` steps
    T0 = 0.10                                  # initial temperature (acc. ~ scale of accuracy deltas)
    T_end = 1e-3
    alpha = (T_end / T0) ** (1.0 / max(1, budget - 1))

    # initial point (random within box), counts as first evaluation
    cur = lo + rng.rand(2) * span
    f_cur = obj.evaluate(cur[0], cur[1])
    best = cur.copy(); f_best = f_cur

    T = T0
    while obj.n_eval < budget:
        cand = cur + rng.normal(0.0, 1.0, size=2) * step
        cand = np.clip(cand, lo, hi)
        f_cand = obj.evaluate(cand[0], cand[1])
        delta = f_cand - f_cur
        if delta >= 0 or rng.rand() < np.exp(delta / max(T, 1e-9)):
            cur, f_cur = cand, f_cand
            if f_cand > f_best:
                best, f_best = cand.copy(), f_cand
        T *= alpha


RUNNERS = {"grid": run_grid, "random": run_random, "bo_tpe": run_bo_tpe,
           "pso": run_pso, "de": run_de, "ga": run_ga, "sa": run_sa}


# =============================================================================
# DATA
# =============================================================================
def load_dataset(openml_id):
    """Load an OpenML dataset and return (X, y) as float arrays.

    Categorical features are ONE-HOT ENCODED (not dropped). Many UCI/OpenML
    classification sets (e.g. 'car') are entirely categorical; dropping
    non-numeric columns would leave zero features. We keep numeric columns
    as-is and one-hot the rest, then concatenate.

    Two feature-count notions are returned via dataset_meta:
      - encoded width (actual model input dimensionality)
      - we also stash the ORIGINAL column count on the array for meta reporting.
    """
    import openml
    from sklearn.preprocessing import OneHotEncoder

    ds = openml.datasets.get_dataset(openml_id)
    Xdf, y, _, _ = ds.get_data(target=ds.default_target_attribute,
                               dataset_format="dataframe")

    n_orig_cols = Xdf.shape[1]
    num = Xdf.select_dtypes(include=[np.number])
    cat = Xdf.select_dtypes(exclude=[np.number])

    parts = []
    if num.shape[1] > 0:
        parts.append(num.values.astype(float))
    if cat.shape[1] > 0:
        # fill missing categoricals with a sentinel so encoder is happy
        cat = cat.astype("object").where(cat.notna(), other="__missing__")
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        parts.append(enc.fit_transform(cat))

    if not parts:
        raise ValueError(f"OpenML {openml_id}: no usable features found.")

    X = np.hstack(parts).astype(float)
    y = pd.factorize(y)[0]
    # attach original column count for meta (encoded width != original width)
    X = np.ascontiguousarray(X)
    load_dataset._last_orig_cols = n_orig_cols
    return X, y


def dataset_meta(X, y):
    n, d = X.shape
    orig = getattr(load_dataset, "_last_orig_cols", d)
    return {"n_samples": int(n),
            "n_features": int(orig),          # original (pre-encoding) features
            "n_features_encoded": int(d),      # actual model input width
            "n_classes": int(len(np.unique(y))),
            "ratio_nd": float(n / orig) if orig else float("nan")}


def maybe_subsample(X, y, cfg, seed):
    cap = cfg.get("max_train_n")
    if cap is None or len(X) <= cap:
        return X, y
    from sklearn.model_selection import train_test_split as tts
    Xs, _, ys, _ = tts(X, y, train_size=cap, random_state=seed, stratify=y)
    return Xs, ys


# =============================================================================
# ONE DATASET  (with checkpoint)
# =============================================================================
def evaluate_test(X_tr, y_tr, X_te, y_te, log2C, log2g, cfg):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="mean")),
        ("sc", StandardScaler()),
        ("svc", SVC(C=2.0**log2C, gamma=2.0**log2g, kernel="rbf")),
    ])
    pipe.fit(X_tr, y_tr)
    return accuracy_score(y_te, pipe.predict(X_te))


def run_one_dataset(oid, name, cfg, optimizers, n_seeds, verbose=True):
    """Returns (rows_df, curves_df, meta_dict). Writes checkpoint files.

    Incremental resume: if a checkpoint exists, only the optimizers NOT yet
    present in it are computed and appended. This allows adding a new optimizer
    (e.g. 'sa') to an already-completed run without recomputing the others.
    """
    ckpt = cfg["checkpoint_dir"]
    os.makedirs(ckpt, exist_ok=True)
    sum_path = os.path.join(ckpt, f"{name}.csv")
    cur_path = os.path.join(ckpt, f"{name}_curves.csv")
    meta_path = os.path.join(ckpt, f"{name}_meta.csv")

    have_ckpt = (os.path.exists(sum_path) and os.path.exists(cur_path)
                 and os.path.exists(meta_path))
    existing_rows = existing_curves = None
    done_opts = set()
    if have_ckpt:
        existing_rows = pd.read_csv(sum_path)
        existing_curves = pd.read_csv(cur_path)
        meta = pd.read_csv(meta_path).iloc[0].to_dict()
        done_opts = set(existing_rows["optimizer"].unique())

    todo = [o for o in optimizers if o not in done_opts]
    if have_ckpt and not todo:
        if verbose:
            print(f"  [skip] {name}: all optimizers present in checkpoint.", flush=True)
        return existing_rows, existing_curves, meta

    X, y = load_dataset(oid)
    meta = dataset_meta(X, y)
    if verbose:
        if done_opts:
            print(f"  {name}: checkpoint has {sorted(done_opts)}; "
                  f"computing only {todo}", flush=True)
        else:
            print(f"  {name}: n={meta['n_samples']} d={meta['n_features']} "
                  f"classes={meta['n_classes']}", flush=True)

    rows, curves = [], []
    t_ds = time.time()
    for seed in range(n_seeds):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=cfg["test_size"], random_state=seed, stratify=y)
        X_tr, y_tr = maybe_subsample(X_tr, y_tr, cfg, seed)

        for opt in todo:
            obj = Objective(X_tr, y_tr, cfg)
            t0 = time.time()
            RUNNERS[opt](obj, cfg, seed)
            elapsed = time.time() - t0
            bc = obj.best_config()
            if bc is None:
                continue
            _, bC, bg, cv_best = bc
            test_acc = evaluate_test(X_tr, y_tr, X_te, y_te, bC, bg, cfg)
            curve = obj.best_so_far_curve()
            thr = 0.99 * cv_best
            reach = int(np.argmax(curve >= thr) + 1) if np.any(curve >= thr) else cfg["budget"]
            rows.append({"dataset": name, "optimizer": opt, "seed": seed,
                         "cv_best": cv_best, "test_acc": test_acc,
                         "best_log2C": bC, "best_log2gamma": bg,
                         "time_sec": elapsed, "evals_to_99pct": reach,
                         "n_evals_used": obj.n_eval})
            for i, v in enumerate(curve):
                curves.append({"dataset": name, "optimizer": opt, "seed": seed,
                               "eval": i + 1, "best_so_far": v})
        if verbose and (seed + 1) % max(1, n_seeds // 5) == 0:
            print(f"    {name}: seed {seed+1}/{n_seeds} "
                  f"({(time.time()-t_ds)/60:.1f} min elapsed)", flush=True)

    dfr = pd.DataFrame(rows)
    dfc = pd.DataFrame(curves)
    # merge with any existing checkpoint content
    if existing_rows is not None:
        dfr = pd.concat([existing_rows, dfr], ignore_index=True)
        dfc = pd.concat([existing_curves, dfc], ignore_index=True)
    dfm = pd.DataFrame([{"dataset": name, "openml_id": oid, **meta}])
    dfr.to_csv(sum_path, index=False)
    dfc.to_csv(cur_path, index=False)
    dfm.to_csv(meta_path, index=False)
    if verbose:
        print(f"  [done] {name} in {(time.time()-t_ds)/60:.1f} min "
              f"-> checkpoint updated ({sorted(set(dfr.optimizer.unique()))}).", flush=True)
    return dfr, dfc, dfm.iloc[0].to_dict()


# =============================================================================
# MAIN
# =============================================================================
def run_all(cfg, datasets=None, optimizers=None, n_seeds=None, verbose=True):
    os.makedirs(cfg["results_dir"], exist_ok=True)
    datasets = datasets if datasets is not None else cfg["datasets"]
    optimizers = optimizers if optimizers is not None else OPTIMIZERS
    n_seeds = n_seeds if n_seeds is not None else cfg["n_seeds"]

    all_rows, all_curves, all_meta = [], [], []
    t0 = time.time()
    for (oid, name) in datasets:
        if verbose:
            print(f"\n=== Dataset: {name} (OpenML {oid}) ===", flush=True)
        dfr, dfc, meta = run_one_dataset(oid, name, cfg, optimizers, n_seeds, verbose)
        all_rows.append(dfr)
        all_curves.append(dfc)
        all_meta.append(meta)
        if verbose:
            print(f"  cumulative wall-clock: {(time.time()-t0)/60:.1f} min", flush=True)

    df = pd.concat(all_rows, ignore_index=True)
    dfc = pd.concat(all_curves, ignore_index=True)
    dfm = pd.DataFrame(all_meta)
    df.to_csv(os.path.join(cfg["results_dir"], "summary.csv"), index=False)
    dfc.to_csv(os.path.join(cfg["results_dir"], "anytime_curves.csv"), index=False)
    dfm.to_csv(os.path.join(cfg["results_dir"], "dataset_meta.csv"), index=False)
    if verbose:
        print(f"\nMerged final outputs written to {cfg['results_dir']}/")
        print(f"Total wall-clock: {(time.time()-t0)/60:.1f} min")
    return df, dfc, dfm


if __name__ == "__main__":
    run_all(CONFIG)
