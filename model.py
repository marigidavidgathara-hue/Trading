"""
model.py
Per-horizon prediction engine. Mirrors the MetaLearner pattern from the
football system: a couple of differently-biased base classifiers, blended,
then calibrated -- rather than trusting one model's raw output.

Two base learners:
  - LightGBM      : captures nonlinear interactions, handles the messy
                     feature distributions technical indicators produce
  - Logistic Reg  : linear, heavily regularized, stabilizes the blend
                     when LightGBM overfits a noisy regime

Blend: simple average of class probabilities (kept deliberately simple --
add a learned meta-weight later if you have enough out-of-fold data to
justify it, same lesson as the football system's MetaLearner C=0.05 fix).

Calibration: isotonic regression per class, fit on a held-out slice, same
CalibratedPair-style wrapper idea as Football_prediction.py -- the raw
blended probabilities are not trustworthy as probabilities until calibrated.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from config import Horizon, MODEL_DIR

CLASSES = [0, 1, 2]  # down, flat, up


class CalibratedPair:
    """One isotonic calibrator per class, applied to a multiclass
    probability vector and renormalized to sum to 1. Same idea as the
    football codebase's CalibratedPair: fit IsotonicRegression directly
    rather than relying on CalibratedClassifierCV(cv='prefit'), which was
    removed in scikit-learn 1.9.
    """

    def __init__(self):
        self.calibrators = {c: IsotonicRegression(out_of_bounds="clip") for c in CLASSES}

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray):
        for c in CLASSES:
            binary_target = (y_true == c).astype(float)
            self.calibrators[c].fit(raw_probs[:, c], binary_target)
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        out = np.column_stack([self.calibrators[c].predict(raw_probs[:, c]) for c in CLASSES])
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return out / row_sums


class DirectionModel:
    """Trains, blends, calibrates, predicts. One instance per (symbol, horizon)."""

    def __init__(self, horizon: Horizon):
        self.horizon = Horizon(horizon)
        self.scaler = StandardScaler()
        self.lgb_model = None
        self.logit_model = None
        self.calibrator = CalibratedPair()
        self.feature_names_: list[str] | None = None

    # -- training -----------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series, calib_frac: float = 0.2, random_state: int = 42):
        self.feature_names_ = list(X.columns)

        # time-ordered split: train -> calibration holdout, no shuffling,
        # since shuffling time series for calibration leaks future info
        # into "past" calibration rows (the exact leakage pattern the
        # football system's audit had to fix).
        n = len(X)
        split = int(n * (1 - calib_frac))
        X_train, X_calib = X.iloc[:split], X.iloc[split:]
        y_train, y_calib = y.iloc[:split], y.iloc[split:]

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_calib_scaled = self.scaler.transform(X_calib)

        # NB: scikit-learn 1.7 removed the `multi_class` kwarg -- lbfgs (the
        # default solver) now always fits a true multinomial model for >2
        # classes, so nothing else needs to change here.
        self.logit_model = LogisticRegression(
            C=0.1, max_iter=2000, class_weight="balanced"
        )
        self.logit_model.fit(X_train_scaled, y_train)

        if _HAS_LGB:
            self.lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                num_leaves=15,
                learning_rate=0.05,
                min_child_samples=30,
                class_weight="balanced",
                verbosity=-1,
            )
            self.lgb_model.fit(X_train, y_train)

        raw_calib_probs = self._blend_raw(X_calib, X_calib_scaled)
        self.calibrator.fit(raw_calib_probs, y_calib.to_numpy())
        return self

    def _blend_raw(self, X_raw: pd.DataFrame, X_scaled: np.ndarray) -> np.ndarray:
        logit_probs = self.logit_model.predict_proba(X_scaled)
        if _HAS_LGB and self.lgb_model is not None:
            lgb_probs = self.lgb_model.predict_proba(X_raw)
            return (logit_probs + lgb_probs) / 2.0
        return logit_probs

    # -- inference -----------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_ is not None:
            X = X[self.feature_names_]
        X_scaled = self.scaler.transform(X)
        raw = self._blend_raw(X, X_scaled)
        calibrated = self.calibrator.transform(raw)
        return pd.DataFrame(calibrated, index=X.index, columns=["p_down", "p_flat", "p_up"])

    def predict_direction(self, X: pd.DataFrame) -> pd.DataFrame:
        probs = self.predict_proba(X)
        direction = probs.idxmax(axis=1).map({"p_down": "DOWN", "p_flat": "FLAT", "p_up": "UP"})
        confidence = probs.max(axis=1)
        return pd.DataFrame({"direction": direction, "confidence": confidence}, index=X.index)

    # -- evaluation (in-sample-of-the-calibration-holdout sanity check) ------
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        probs = self.predict_proba(X)
        y_arr = y.to_numpy()
        preds = probs.to_numpy().argmax(axis=1)
        brier = np.mean([
            brier_score_loss((y_arr == c).astype(int), probs.iloc[:, c])
            for c in range(3)
        ])
        return {
            "accuracy": accuracy_score(y_arr, preds),
            "log_loss": log_loss(y_arr, probs.to_numpy(), labels=CLASSES),
            "brier_score_avg": brier,
        }

    # -- persistence ----------------------------------------------------------
    def save(self, symbol: str):
        path = self._path_for(symbol, self.horizon)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def _path_for(symbol: str, horizon: Horizon) -> Path:
        safe_symbol = symbol.replace("/", "_").replace("=", "_")
        return MODEL_DIR / f"{safe_symbol}_{horizon.value}.pkl"

    @classmethod
    def load(cls, symbol: str, horizon: Horizon) -> "DirectionModel":
        path = cls._path_for(symbol, Horizon(horizon))
        with open(path, "rb") as f:
            return pickle.load(f)
