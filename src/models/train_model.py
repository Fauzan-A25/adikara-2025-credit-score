"""Stratified K-Fold berulang, LightGBM, dan pemilihan ambang untuk macro F1."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

LGB_PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8)
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


def lgb_model(seed: int = 0, **kw):
    return lgb.LGBMClassifier(**{**LGB_PARAMS, **kw}, random_state=seed, verbose=-1)


def cv_predict(make_model, X, y, seeds=range(5), n_splits: int = 5) -> np.ndarray:
    """Probabilitas out-of-fold untuk tiap seed. Hasil: array (n_seed, n_baris)."""
    y = np.asarray(y)
    out = np.zeros((len(list(seeds)), len(y)))
    for i, s in enumerate(seeds):
        for trn, val in StratifiedKFold(n_splits, shuffle=True, random_state=s).split(X, y):
            out[i, val] = make_model(s).fit(X.iloc[trn], y[trn]).predict_proba(X.iloc[val])[:, 1]
    return out


def f1_curve(y, p) -> np.ndarray:
    return np.array([f1_score(y, p >= t, average="macro") for t in THRESHOLDS])


def best_threshold(y, p) -> tuple[float, float]:
    c = f1_curve(y, p)
    i = int(np.argmax(c))
    return float(THRESHOLDS[i]), float(c[i])


def macro_f1_by_seed(oof: np.ndarray, y, mask=None) -> tuple[list[float], list[float]]:
    """Macro F1 terbaik (ambang dipilih per seed) dan ambangnya, pada baris `mask`."""
    y = np.asarray(y)
    mask = np.ones(len(y), bool) if mask is None else np.asarray(mask)
    res = [best_threshold(y[mask], p[mask]) for p in np.atleast_2d(oof)]
    return [r[1] for r in res], [r[0] for r in res]
