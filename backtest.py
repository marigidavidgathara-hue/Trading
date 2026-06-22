"""
backtest.py
Walk-forward backtest harness -- same spirit as the football system's
backtest_mc_vs_analytical.py: don't trust a single train/test split,
roll the window forward repeatedly and aggregate out-of-sample predictions
before judging anything.

Reports:
  - accuracy / log loss / Brier score on pooled out-of-sample predictions
  - a calibration table (is "70% confident" actually right ~70% of the time?)
  - a naive equity curve IF you traded every prediction above a confidence
    threshold, net of a per-trade cost assumption

This backtest tells you whether the model's probabilities are honest and
whether the historical edge (if any) survives costs. It does NOT tell you
the strategy will keep working going forward -- markets are non-stationary
in a way Brasileirão fixtures aren't.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

from features import build_features, build_labels
from model import DirectionModel, CLASSES
from config import Horizon


def walk_forward_predictions(
    df: pd.DataFrame,
    horizon: Horizon,
    lookahead: int,
    train_window: int,
    test_window: int,
    has_volume: bool = True,
) -> pd.DataFrame:
    """Slides a fixed-size training window forward in `test_window`-sized
    steps, refitting each time, and predicts on the immediately-following
    untouched slice. Returns one row per out-of-sample bar with the
    calibrated probabilities, the model's pick, and the realized label.
    """
    feats = build_features(df, has_volume=has_volume)
    labels = build_labels(df, lookahead).reindex(feats.index)
    valid = labels.notna()
    feats, labels = feats[valid], labels[valid].astype(int)

    rows = []
    start = 0
    n = len(feats)
    while start + train_window + test_window <= n:
        train_slice = slice(start, start + train_window)
        test_slice = slice(start + train_window, start + train_window + test_window)

        X_train, y_train = feats.iloc[train_slice], labels.iloc[train_slice]
        X_test, y_test = feats.iloc[test_slice], labels.iloc[test_slice]

        model = DirectionModel(horizon)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)
        probs["y_true"] = y_test.to_numpy()
        rows.append(probs)

        start += test_window

    if not rows:
        raise ValueError(
            "Not enough data for even one walk-forward fold -- "
            "reduce train_window/test_window or fetch more history."
        )
    return pd.concat(rows)


def score_predictions(pooled: pd.DataFrame) -> dict:
    y_true = pooled["y_true"].to_numpy()
    probs = pooled[["p_down", "p_flat", "p_up"]].to_numpy()
    preds = probs.argmax(axis=1)

    brier = np.mean([
        brier_score_loss((y_true == c).astype(int), probs[:, i])
        for i, c in enumerate(CLASSES)
    ])
    return {
        "n_predictions": len(pooled),
        "accuracy": accuracy_score(y_true, preds),
        "log_loss": log_loss(y_true, probs, labels=CLASSES),
        "brier_score_avg": brier,
    }


def calibration_table(pooled: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table: bin predictions by the model's own stated
    confidence (max class probability), and check whether the model was
    actually right that often within each bin. A well-calibrated model
    has mean_confidence ~= empirical_accuracy in every row."""
    probs = pooled[["p_down", "p_flat", "p_up"]]
    confidence = probs.max(axis=1)
    predicted_class = probs.to_numpy().argmax(axis=1)
    correct = (predicted_class == pooled["y_true"].to_numpy()).astype(int)

    bins = pd.cut(confidence, bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    table = pd.DataFrame({"confidence": confidence, "correct": correct, "bin": bins})
    summary = table.groupby("bin", observed=True).agg(
        mean_confidence=("confidence", "mean"),
        empirical_accuracy=("correct", "mean"),
        n=("correct", "size"),
    )
    return summary


def simulate_equity_curve(
    pooled: pd.DataFrame,
    df: pd.DataFrame,
    lookahead: int,
    confidence_threshold: float = 0.5,
    cost_per_trade: float = 0.0005,
) -> pd.DataFrame:
    """Naive simulation: when the model's top probability clears
    `confidence_threshold` and that class isn't FLAT, take a directional
    position sized 1.0, hold for `lookahead` bars, realize the actual
    forward return minus `cost_per_trade` (round-trip cost as a fraction,
    e.g. 0.0005 = 5 bps -- set this to your real fees + expected slippage).
    Everything else sits in cash for that bar. This is a sanity check on
    whether the edge survives costs, not a production execution model --
    it ignores position overlap, sizing, and slippage beyond the flat fee.
    """
    fwd_ret = (df["close"].shift(-lookahead) / df["close"] - 1).reindex(pooled.index)
    probs = pooled[["p_down", "p_flat", "p_up"]]
    confidence = probs.max(axis=1)
    direction = probs.to_numpy().argmax(axis=1)  # 0 down, 1 flat, 2 up

    take_trade = (confidence >= confidence_threshold) & (direction != 1)
    side = np.where(direction == 2, 1, -1)  # +1 long, -1 short

    trade_pnl = np.where(take_trade, side * fwd_ret.to_numpy() - cost_per_trade, 0.0)
    curve = pd.DataFrame({
        "pnl": trade_pnl,
        "traded": take_trade,
    }, index=pooled.index)
    curve["equity"] = (1 + curve["pnl"]).cumprod()
    return curve


def run_backtest_report(
    df: pd.DataFrame,
    horizon: Horizon,
    lookahead: int,
    train_window: int,
    test_window: int,
    has_volume: bool = True,
    confidence_threshold: float = 0.5,
    cost_per_trade: float = 0.0005,
) -> dict:
    """One-call convenience wrapper used by main.py's --backtest flag."""
    pooled = walk_forward_predictions(df, horizon, lookahead, train_window, test_window, has_volume)
    scores = score_predictions(pooled)
    calib = calibration_table(pooled)
    equity = simulate_equity_curve(pooled, df, lookahead, confidence_threshold, cost_per_trade)
    n_trades = int(equity["traded"].sum())
    return {
        "scores": scores,
        "calibration_table": calib,
        "equity_curve": equity,
        "n_trades": n_trades,
        "final_equity": float(equity["equity"].iloc[-1]) if len(equity) else 1.0,
        "win_rate_on_traded": float((equity.loc[equity["traded"], "pnl"] > 0).mean()) if n_trades else float("nan"),
    }
