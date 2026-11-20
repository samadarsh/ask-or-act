"""Classic ML baseline (type ``classic``). Free, needs the ``classic`` extra (scikit-learn).

* **Banking77 and Aegis:** TF-IDF (word 1–2-grams, sublinear tf) + logistic regression.
* **MT-Bench:** there's no text to classify between two answers, so a small feature
  model on answer lengths is used. It's a deliberately weak "verbosity heuristic" that
  the LLM judges have to beat.

**Calibrated probabilities: what the temperature is fitted on.**
``CalibratedClassifierCV(cv=5)`` with **temperature scaling** (softmax(logits / T)). The
*training* data is split into 5 folds. For each fold, the features + logistic regression
are fitted on the other 4, and T is fitted on the held-out fold's predictions. The final
model averages the 5 (model, T) pairs. So T is never fitted on test data, and for
Banking77/Aegis never on our sample's calibration half either. The feature step (TF-IDF
vocabulary/IDF, or the scaler) sits *inside* each fold, so the fold that fits T never
shaped the features.

Temperature scaling rather than Platt/isotonic: those calibrate each class one-vs-rest
and then renormalize, which squashes multiclass probabilities. This was observed on
Banking77: sigmoid gave mean confidence 0.66 at accuracy 0.86 (ECE 0.21). Chosen for that
reason, not by comparing test scores.

**MT-Bench cross-fitting.** MT-Bench has no official training split, so the model trains
on our sample's calibration half. Milestone 3b fits thresholds on that same half, so it
needs honest (out-of-sample) predictions there. The calibration half is split into
``cross_fit`` folds **by question** (a question's comparisons never straddle folds). Each
calibration example is predicted by a model trained on the other folds, and
``raw["cross_fit_fold"]`` records which fold. Test examples use the model trained on the
whole calibration half. Inner calibration folds are grouped by question too.

The model is fitted lazily on the first batch, so a dry run needs neither data nor
scikit-learn. Training is deterministic (fixed seeds, deterministic folds), so one config
always gives the same model and the cache stays valid.
"""

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ask_or_act.core.decider import CostEstimate
from ask_or_act.core.types import JSON, DeciderConfig, Decision, PublicExample
from ask_or_act.deciders import DeciderContext

DEFAULTS: dict[str, JSON] = {
    "C": 4.0,
    "ngram_max": 2,
    "min_df": 2,
    "cv": 5,
    "seed": 0,
    "calibration": "temperature",
    "cross_fit": 5,  # outer folds for grouped training sets (MT-Bench); 0 = off
    "features": "v2",  # v2: features fitted inside each calibration fold
}


def _turn_answers(inp: JSON) -> tuple[str, str]:
    if not isinstance(inp, dict):
        raise ValueError("MT-Bench input must be a dict")
    turn = int(str(inp["turn"]))
    conv_a, conv_b = inp["conversation_a"], inp["conversation_b"]
    if not isinstance(conv_a, list) or not isinstance(conv_b, list):
        raise ValueError("conversations must be lists")
    pick = 2 * turn - 1  # the assistant message of the judged turn

    def content(conv: list[JSON]) -> str:
        msg = conv[pick]
        return str(msg["content"]) if isinstance(msg, dict) else ""

    return content(conv_a), content(conv_b)


def length_features(inputs: Sequence[JSON]) -> np.ndarray:
    rows = []
    for inp in inputs:
        a, b = _turn_answers(inp)
        la, lb = len(a), len(b)
        rows.append([math.log1p(la), math.log1p(lb), math.log1p(la) - math.log1p(lb)])
    return np.array(rows)


def group_folds(groups: Sequence[str], k: int, seed: int) -> list[int]:
    """Fold number per row. Whole groups go to one fold. Groups are put in a seeded
    hash order and dealt round-robin, so folds get (nearly) equal numbers of groups.
    Deterministic: same groups and seed, same folds."""
    unique = sorted(set(groups), key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).digest())
    if len(unique) < k:
        raise ValueError(f"need at least {k} groups for {k} folds, got {len(unique)}")
    fold_of = {g: i % k for i, g in enumerate(unique)}
    return [fold_of[g] for g in groups]


def _splits(folds: Sequence[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    arr = np.asarray(folds)
    return [(np.flatnonzero(arr != f), np.flatnonzero(arr == f)) for f in sorted(set(folds))]


class ClassicDecider:
    def __init__(self, config: DeciderConfig, ctx: DeciderContext) -> None:
        unknown = set(config.params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown classic params: {sorted(unknown)}")
        self.params = {**DEFAULTS, **config.params}
        # Report resolved defaults, so changing a default changes the cache key.
        self._config = DeciderConfig(
            config.type, config.model, config.prompt_version, dict(self.params)
        )
        self._ctx = ctx
        self._model: Any = None
        self._classes: list[str] = []
        self._train_ids: frozenset[str] = frozenset()
        self._fold_of: dict[str, int] = {}  # training id -> cross-fit fold
        self._fold_models: list[Any] = []
        self.training_description = ""

    @property
    def config(self) -> DeciderConfig:
        return self._config

    @property
    def max_batch_size(self) -> int:
        return 1_000

    @property
    def needs_gold(self) -> bool:
        return False

    def estimate(self, examples: Sequence[PublicExample]) -> CostEstimate:
        return CostEstimate(1, 0, 0, 0.0)

    def _x(self, inputs: Sequence[JSON]) -> list[Any]:
        return list(inputs) if self._ctx.dataset == "mt_bench" else [str(i) for i in inputs]

    def _estimator(self, cv: Any) -> Any:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer, StandardScaler

        if self._ctx.dataset == "mt_bench":
            features: Any = make_pipeline(FunctionTransformer(length_features), StandardScaler())
        else:
            features = TfidfVectorizer(
                ngram_range=(1, int(str(self.params["ngram_max"]))),
                min_df=int(str(self.params["min_df"])),
                sublinear_tf=True,
            )
        lr = LogisticRegression(
            C=float(str(self.params["C"])),
            max_iter=2000,
            random_state=int(str(self.params["seed"])),
        )
        # Features inside the calibrated estimator: each fold fits its own features.
        return CalibratedClassifierCV(
            make_pipeline(features, lr), method=str(self.params["calibration"]), cv=cv
        )

    def _fit_one(self, x: list[Any], labels: list[str], groups: list[str] | None) -> Any:
        k = int(str(self.params["cv"]))
        cv: Any = (
            k if groups is None else _splits(group_folds(groups, k, int(str(self.params["seed"]))))
        )
        return self._estimator(cv).fit(x, labels)

    def fit(
        self,
        ids: Sequence[str],
        inputs: Sequence[JSON],
        labels: Sequence[str],
        groups: Sequence[str] | None = None,
    ) -> None:
        """Fit on training data. With ``groups`` (MT-Bench question ids) the inner
        calibration folds are grouped, and ``cross_fit`` > 0 also fits one model per
        outer fold for out-of-fold predictions on the training ids themselves."""
        x, y = self._x(inputs), list(labels)
        g = None if groups is None else [str(v) for v in groups]
        self._model = self._fit_one(x, y, g)
        self._classes = [str(c) for c in self._model.classes_]
        self._train_ids = frozenset(ids)
        self._fold_of, self._fold_models = {}, []
        k_outer = int(str(self.params["cross_fit"]))
        if g is not None and k_outer >= 2:
            folds = group_folds(g, k_outer, int(str(self.params["seed"])) + 1)
            for f in range(k_outer):
                rows = [i for i, fi in enumerate(folds) if fi != f]
                model = self._fit_one(
                    [x[i] for i in rows], [y[i] for i in rows], [g[i] for i in rows]
                )
                if [str(c) for c in model.classes_] != self._classes:
                    raise ValueError(f"cross-fit fold {f} is missing a class; use fewer folds")
                self._fold_models.append(model)
            self._fold_of = dict(zip(ids, folds, strict=True))

    def _ensure_fitted(self) -> None:
        if self._model is None:
            from ask_or_act.data import training

            ts = training.load(self._ctx.dataset, self._ctx.raw_dir)
            self.training_description = ts.description
            self.fit(ts.ids, ts.inputs, ts.labels, ts.groups)

    def _proba(self, examples: Sequence[PublicExample]) -> list[tuple[np.ndarray, int | None]]:
        """Probability rows, plus the cross-fit fold used (None = full model)."""
        out: list[tuple[np.ndarray, int | None]] = [(np.empty(0), None)] * len(examples)
        by_model: dict[int | None, list[int]] = {}
        for i, ex in enumerate(examples):
            if ex.id in self._train_ids:
                if ex.id not in self._fold_of:
                    raise ValueError(f"refusing to predict on training examples: {ex.id}")
                by_model.setdefault(self._fold_of[ex.id], []).append(i)
            else:
                by_model.setdefault(None, []).append(i)
        for fold, idx in by_model.items():
            model = self._model if fold is None else self._fold_models[fold]
            rows = model.predict_proba(self._x([examples[i].input for i in idx]))
            for i, row in zip(idx, rows, strict=True):
                out[i] = (row, fold)
        return out

    async def decide_batch(self, examples: Sequence[PublicExample]) -> list[Decision]:
        self._ensure_fitted()
        out = []
        for ex, (row, fold) in zip(examples, self._proba(examples), strict=True):
            dist = dict.fromkeys(ex.options, 0.0)
            for cls, p in zip(self._classes, row, strict=True):
                if cls in dist:
                    dist[cls] = float(p)
            total = sum(dist.values()) or 1.0
            dist = {o: p / total for o, p in dist.items()}
            choice = max(ex.options, key=lambda o: dist[o])
            raw: dict[str, JSON] = {"training": self.training_description}
            if fold is not None:
                raw["cross_fit_fold"] = fold
            out.append(
                Decision(
                    ex.id,
                    choice,
                    dist[choice],
                    probabilities=dist,
                    signals={"native": dist[choice]},
                    model=self._config.model,
                    raw=raw,
                )
            )
        return out
