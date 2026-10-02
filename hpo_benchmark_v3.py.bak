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
    # Variance-decomposition mode (Reviewer R1.11 / R2.2). One of:
    #   "baseline"   -> reproduce the submitted wiring EXACTLY
    #                   (split & optimizer share rep; CV folds fixed)
    #   "isolate_F"  -> vary CV folds only        (true CV noise)
    #   "isolate_S"  -> vary train/test split only
    #   "isolate_O"  -> vary optimizer init only
    "decomp_mode": "baseline",
    # Grid traversal order (Reviewer R2.6): "random" (default, averages over
    # orderings), "row_major" (the submitted behaviour), "col_major", "spiral".
    "grid_order": "row_major",
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
    def __init__(self, X, y, cfg, cv_seed=None):
        self.X = X
        self.y = y
        self.cfg = cfg
        # cv_seed defaults to the config value so existing callers are unaffected;
        # the decomposition passes an explicit per-repetition value.
        cv_seed = cfg["cv_seed"] if cv_seed is None else cv_seed
        self.cv = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True,
                                  random_state=cv_seed)
        self.history = []
        self.n_eval = 0
        # Weighted budget accounting (fold-fit budget definition, Stage #2).
        # For full-fidelity optimizers (frac=1) `spent` increments by 1.0 per call,
        # so it is numerically identical to n_eval and the budget guard is unchanged.
        # Multi-fidelity methods (e.g. successive halving) pass frac<1, so partial-
        # data evaluations cost proportionally less.
        self.spent = 0.0

    def evaluate(self, log2C, log2g, frac=1.0):
        # cost of this evaluation in full-fidelity-equivalent units
        p = self.cfg.get("fidelity_exponent", 1.0)
        cost = float(frac) ** p
        if self.spent + cost > self.cfg["budget"] + 1e-9:
            return max((h[3] for h in self.history), default=0.0)
        # For frac<1, evaluate on a stratified subsample of that fraction; for the
        # default frac=1.0 the full data is used and behaviour is unchanged.
        if frac >= 1.0:
            Xe, ye = self.X, self.y
        else:
            Xe, ye = self._subsample(frac)
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("sc", StandardScaler()),
            ("svc", SVC(C=2.0 ** log2C, gamma=2.0 ** log2g, kernel="rbf")),
        ])
        score = cross_val_score(pipe, Xe, ye, cv=self.cv,
                                scoring="accuracy",
                                n_jobs=self.cfg.get("cv_n_jobs", 1)).mean()
        self.n_eval += 1
        self.spent += cost
        # store (raw_index, log2C, log2g, score, weighted_spent_after)
        self.history.append((self.n_eval, log2C, log2g, score, self.spent))
        return score

    def _subsample(self, frac):
        """Stratified subsample of the training data to a given fraction, used only
        by multi-fidelity optimizers. Deterministic given cfg['cv_seed'] so that
        fidelity rungs are reproducible."""
        from sklearn.model_selection import train_test_split as _tts
        n = len(self.y)
        k = max(self.cfg["cv_folds"] + 1, int(round(frac * n)))
        if k >= n:
            return self.X, self.y
        Xs, _, ys, _ = _tts(self.X, self.y, train_size=k,
                            random_state=self.cfg["cv_seed"], stratify=self.y)
        return Xs, ys

    def best_so_far_curve(self):
        """Anytime best-so-far accuracy on the WEIGHTED budget axis.

        The x-axis has cfg['budget'] slots, one per full-fidelity-equivalent unit of
        cost. For full-fidelity optimizers (cost=1 per eval) this is identical to the
        old per-evaluation curve. For multi-fidelity SH, each evaluation advances the
        axis by its fractional cost, so the curve is expressed in the same budget
        units as every other optimizer -- making anytime comparison fair and
        order-independent (Reviewers R2.4 / R2.6)."""
        B = self.cfg["budget"]
        curve = np.full(B, np.nan)
        best = -np.inf
        for h in self.history:
            s = h[3]
            spent_after = h[4] if len(h) > 4 else h[0]
            best = max(best, s)
            slot = int(np.ceil(spent_after)) - 1        # weighted-budget slot
            slot = min(max(slot, 0), B - 1)             # clip into range
            # keep the best value seen up to this slot
            if np.isnan(curve[slot]) or best > curve[slot]:
                curve[slot] = best
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
        # history entries are 5-tuples (idx, log2C, log2g, score, spent); downstream
        # callers expect the original 4-tuple (idx, log2C, log2g, score).
        b = max(self.history, key=lambda h: h[3])
        return (b[0], b[1], b[2], b[3])


# =============================================================================
# OPTIMIZERS
# =============================================================================
def run_grid(obj, cfg, seed):
    """Grid search over a side x side lattice.

    TRAVERSAL ORDER (Reviewer R2.6). A grid has no intrinsic evaluation order: the
    SAME set of lattice points can be visited in any sequence. The submitted version
    used a fixed row-major sweep (C ascending, gamma ascending), which starts in the
    low-C/low-gamma corner -- typically a poor-accuracy region -- and produces the
    characteristic staircase in the anytime curve. Because "evaluations to reach a
    target" depends on when good points are visited, that staircase (and any
    efficiency number derived from it) is partly an ORDERING artefact rather than a
    property of grid search itself.

    cfg["grid_order"] selects the policy:
      "random"     (default) : uniformly random permutation of the lattice, reseeded
                               per repetition -> the reported efficiency is averaged
                               over many orderings, so it is order-independent.
      "row_major"            : the original deterministic sweep (for reproducing the
                               submitted numbers and for the ordering comparison).
      "col_major"            : deterministic sweep with the loops swapped.
      "spiral"               : centre-outwards; a "smart" deterministic policy.

    Reporting the distribution over random orderings, and contrasting it with the
    deterministic policies, shows how much of the grid-search disadvantage is real
    and how much was ordering.
    """
    lc = cfg["log2C_bounds"]; lg = cfg["log2gamma_bounds"]
    side = int(round(cfg["budget"] ** 0.5))
    cs = np.linspace(lc[0], lc[1], side)
    gs = np.linspace(lg[0], lg[1], side)

    # build the full lattice as a list of (c, g) points
    pts = [(c, g) for c in cs for g in gs]          # row-major by construction
    order = cfg.get("grid_order", "random")

    if order == "row_major":
        pass                                         # already row-major
    elif order == "col_major":
        pts = [(c, g) for g in gs for c in cs]
    elif order == "spiral":
        # centre-outwards: sort by distance from the lattice centre
        c0, g0 = (lc[0] + lc[1]) / 2.0, (lg[0] + lg[1]) / 2.0
        pts.sort(key=lambda p: (p[0] - c0) ** 2 + (p[1] - g0) ** 2)
    elif order == "random":
        rng = np.random.RandomState(seed)            # reseeded per repetition
        idx = rng.permutation(len(pts))
        pts = [pts[i] for i in idx]
    else:
        raise ValueError(f"unknown grid_order: {order!r}")

    for c, g in pts:
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


def run_cuckoo(obj, cfg, seed):
    """Cuckoo Search (Yang & Deb, 2009) over the continuous (log2C, log2gamma) space.

    A recent, widely-used population-based metaheuristic driven by Levy-flight
    global exploration plus a discovery-and-abandonment step. Added as a post-2024
    representative optimizer (Reviewers R1.12 / R2.1).

    Fairness (identical to the other seven optimizers):
      * Full-fidelity: every nest is scored by one full k-fold CV on the full data
        (frac=1.0), so one nest evaluation costs exactly one budget unit under the
        fold-fit budget definition -- no special budget handling is required.
      * Identical search box (cfg['log2C_bounds'], cfg['log2gamma_bounds']) and
        identical total budget (cfg['budget'] evaluations).
      * Standard, un-tuned control parameters to avoid advantaging the method:
        n_nests = 15 (same order as the pop_size=20 of PSO/DE/GA),
        discovery rate pa = 0.25, Levy exponent beta = 1.5 (canonical values).

    Reference: X.-S. Yang and S. Deb, "Cuckoo search via Levy flights,"
    World Congress on Nature & Biologically Inspired Computing, 2009.
    """
    import math
    rng = np.random.RandomState(seed)
    lc = cfg["log2C_bounds"]; lg = cfg["log2gamma_bounds"]
    budget = cfg["budget"]
    lo = np.array([lc[0], lg[0]], dtype=float)
    hi = np.array([lc[1], lg[1]], dtype=float)
    span = hi - lo

    n_nests = cfg.get("cuckoo_n_nests", 15)
    pa      = cfg.get("cuckoo_pa", 0.25)      # discovery (abandonment) rate
    beta    = cfg.get("cuckoo_beta", 1.5)     # Levy exponent

    # Mantegna's algorithm for Levy-flight step scaling
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta *
                2 ** ((beta - 1) / 2))) ** (1 / beta)

    def levy_step(size):
        u = rng.normal(0, sigma_u, size=size)
        v = rng.normal(0, 1.0, size=size)
        return u / (np.abs(v) ** (1 / beta))

    def clip(x):
        return np.clip(x, lo, hi)

    def evaluate(x):
        # full-fidelity evaluation; budget is guarded by obj.evaluate itself
        return obj.evaluate(x[0], x[1])

    # --- initialise nests (each initial nest counts as one evaluation) ---
    nests = lo + rng.rand(n_nests, 2) * span
    fitness = np.full(n_nests, -np.inf)
    for i in range(n_nests):
        if obj.n_eval >= budget:
            break
        fitness[i] = evaluate(nests[i])

    best_idx = int(np.argmax(fitness))
    best = nests[best_idx].copy()
    f_best = fitness[best_idx]

    # --- main loop ---
    while obj.n_eval < budget:
        # (1) Levy-flight: generate a new solution from a random nest, move toward best
        i = rng.randint(n_nests)
        step = 0.01 * levy_step(2) * (nests[i] - best)
        cand = clip(nests[i] + step * rng.normal(size=2))
        if obj.n_eval >= budget:
            break
        f_cand = evaluate(cand)
        # replace a randomly chosen nest if the new solution is better
        j = rng.randint(n_nests)
        if f_cand > fitness[j]:
            nests[j] = cand
            fitness[j] = f_cand
            if f_cand > f_best:
                best, f_best = cand.copy(), f_cand

        # (2) abandon a fraction pa of worst nests, build new ones (biased random walk)
        n_abandon = int(pa * n_nests)
        if n_abandon > 0:
            worst = np.argsort(fitness)[:n_abandon]
            for w in worst:
                if obj.n_eval >= budget:
                    break
                d1 = nests[rng.randint(n_nests)]
                d2 = nests[rng.randint(n_nests)]
                new = clip(nests[w] + rng.rand() * (d1 - d2))
                f_new = evaluate(new)
                if f_new > fitness[w]:
                    nests[w] = new
                    fitness[w] = f_new
                    if f_new > f_best:
                        best, f_best = new.copy(), f_new


def run_sh(obj, cfg, seed):
    """Successive Halving over the (log2C, log2gamma) space (Jamieson & Talwalkar
    2016; Li et al. 2018). Added as a post-2024 multi-fidelity representative
    (Reviewers R1.12 / R2.1).

    Unlike the full-fidelity optimizers, SH allocates a small data fraction to many
    candidates first, then promotes survivors to larger fractions. It therefore
    consumes the budget in fractional units under the fold-fit budget definition:
    a candidate scored on fraction f costs f**p full-fidelity-equivalent units
    (p = cfg['fidelity_exponent']). SH stops when the accumulated weighted cost
    reaches cfg['budget'], exactly like every other optimizer.

    Design (standard, un-tuned, to avoid advantaging SH):
      * eta = 3 (canonical halving factor).
      * Rungs use data fractions f in a geometric ladder ending at f = 1.0.
      * Candidates are sampled uniformly at random in the same search box.
      * Random state derived from `seed` so runs are reproducible.

    This is a self-contained SH so that the SAME Objective (and hence the same CV
    folds, budget accounting, and history) is used as for the other optimizers,
    which keeps the comparison exactly budget-matched. We deliberately do not use
    sklearn's HalvingRandomSearchCV here because it manages its own CV and cannot
    report into our shared budget ledger; the algorithm implemented below is the
    standard successive-halving procedure.
    """
    rng = np.random.RandomState(seed)
    lc = cfg["log2C_bounds"]; lg = cfg["log2gamma_bounds"]
    lo = np.array([lc[0], lg[0]], dtype=float)
    hi = np.array([lc[1], lg[1]], dtype=float)
    span = hi - lo
    budget = cfg["budget"]
    eta = cfg.get("sh_eta", 3)
    min_frac = cfg.get("sh_min_frac", 1.0 / (eta ** 3))   # smallest rung fraction

    # geometric ladder of data fractions ending at 1.0: e.g. 1/27, 1/9, 1/3, 1
    fracs = []
    f = min_frac
    while f < 1.0 - 1e-9:
        fracs.append(f); f *= eta
    fracs.append(1.0)

    # number of initial candidates: enough that the ladder ends with a few finalists,
    # while the whole bracket fits inside the weighted budget. Choose n0 so that the
    # weighted cost of one full bracket is <= budget; scale up brackets until budget
    # is exhausted.
    def bracket_weighted_cost(n0):
        c = 0.0; n = n0
        for fr in fracs:
            c += n * (fr ** cfg.get("fidelity_exponent", 1.0))
            n = max(1, n // eta)
        return c

    n0 = eta ** len(fracs)                    # e.g. 3^4 = 81 candidates at the base
    while bracket_weighted_cost(n0) > budget and n0 > eta:
        n0 = max(eta, n0 // eta)

    def sample(n):
        return lo + rng.rand(n, 2) * span

    def run_bracket():
        cands = sample(n0)
        n = n0
        for fr in fracs:
            scores = []
            for i in range(len(cands)):
                if obj.spent + (fr ** cfg.get("fidelity_exponent", 1.0)) > budget + 1e-9:
                    return
                s = obj.evaluate(cands[i][0], cands[i][1], frac=fr)
                scores.append(s)
            scores = np.array(scores)
            keep = max(1, len(cands) // eta)
            order = np.argsort(scores)[::-1][:keep]
            cands = cands[order]
            n = keep

    # run brackets until the weighted budget is spent
    while obj.spent < budget - 1e-9:
        before = obj.spent
        run_bracket()
        if obj.spent <= before + 1e-9:   # safety: no progress -> stop
            break


RUNNERS = {"grid": run_grid, "random": run_random, "bo_tpe": run_bo_tpe,
           "pso": run_pso, "de": run_de, "ga": run_ga, "sa": run_sa,
           "cuckoo": run_cuckoo, "sh": run_sh}


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


# =============================================================================
# SEED RESOLUTION (variance decomposition: Reviewer R1.11 / R2.2)
# =============================================================================
# Fixed values used when a source is held constant during isolation.
_FIXED = {"split": 0, "opt": 0}  # cv fixed value taken from cfg["cv_seed"]

def _resolve_seeds(rep, cfg):
    """Map a repetition index to (seed_split, seed_cv, seed_opt) per mode.

    "baseline" reproduces the submitted study EXACTLY: the train/test split
    and the optimizer share the same seed (=rep), and the CV folds are fixed
    at cfg["cv_seed"]. The three isolate_* modes vary exactly one source and
    freeze the other two, so each measured variance is attributable to a
    single named source.
    """
    mode = cfg.get("decomp_mode", "baseline")
    cv_fixed = cfg["cv_seed"]
    if mode == "baseline":
        return rep, cv_fixed, rep
    if mode == "isolate_F":            # vary CV folds only -> true CV noise
        return _FIXED["split"], rep, _FIXED["opt"]
    if mode == "isolate_S":            # vary train/test split only
        return rep, cv_fixed, _FIXED["opt"]
    if mode == "isolate_O":            # vary optimizer init only
        return _FIXED["split"], cv_fixed, rep
    raise ValueError(f"unknown decomp_mode: {mode!r}")


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
    for rep in range(n_seeds):
        # Three independent seeds; in "baseline" mode this reproduces the
        # original wiring exactly (seed_split == seed_opt == rep, cv fixed).
        seed_split, seed_cv, seed_opt = _resolve_seeds(rep, cfg)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=cfg["test_size"], random_state=seed_split, stratify=y)
        X_tr, y_tr = maybe_subsample(X_tr, y_tr, cfg, seed_split)

        for opt in todo:
            # Grid search has no optimizer RNG; its seed_opt is irrelevant and
            # Var(O)=0 for it by construction (a built-in sanity check).
            obj = Objective(X_tr, y_tr, cfg, cv_seed=seed_cv)
            t0 = time.time()
            RUNNERS[opt](obj, cfg, seed_opt)
            elapsed = time.time() - t0
            bc = obj.best_config()
            if bc is None:
                continue
            _, bC, bg, cv_best = bc
            test_acc = evaluate_test(X_tr, y_tr, X_te, y_te, bC, bg, cfg)
            curve = obj.best_so_far_curve()
            thr = 0.99 * cv_best
            reach = int(np.argmax(curve >= thr) + 1) if np.any(curve >= thr) else cfg["budget"]
            rows.append({"dataset": name, "optimizer": opt,
                         "rep": rep, "seed": rep,
                         "seed_split": seed_split, "seed_cv": seed_cv,
                         "seed_opt": seed_opt, "decomp_mode": cfg.get("decomp_mode", "baseline"),
                         "cv_best": cv_best, "test_acc": test_acc,
                         "best_log2C": bC, "best_log2gamma": bg,
                         "time_sec": elapsed, "evals_to_99pct": reach,
                         "n_evals_used": obj.n_eval})
            for i, v in enumerate(curve):
                curves.append({"dataset": name, "optimizer": opt,
                               "rep": rep, "seed": rep,
                               "eval": i + 1, "best_so_far": v})
        if verbose and (rep + 1) % max(1, n_seeds // 5) == 0:
            print(f"    {name}: rep {rep+1}/{n_seeds} "
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