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


def _walk_forward_core(
    feats: pd.DataFrame,
    labels: pd.Series,
    horizon: Horizon,
    train_window: int,
    test_window: int,
) -> pd.DataFrame:
    """Slides a fixed-size training window forward in `test_window`-sized
    steps, refitting each time, and predicts on the immediately-following
    untouched slice. Returns one row per out-of-sample bar with the
    calibrated probabilities, the model's pick, and the realized label.
    """
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


# Floor below which a walk-forward fold isn't meaningful -- also the
# minimum resolve_windows will shrink down to when there isn't enough
# history at the configured defaults.
MIN_WINDOW = {
    Horizon.SCALP: dict(train_window=300, test_window=50),
    Horizon.SWING: dict(train_window=100, test_window=20),
}


def resolve_windows(n_available: int, horizon: Horizon, requested: dict) -> dict:
    """If the requested (default) window sizes don't fit the data actually
    available, shrink them proportionally down to MIN_WINDOW's floor
    instead of failing outright. This is what's needed when, e.g., a
    yfinance-backed market returns far fewer bars than crypto's much
    deeper history -- same model/backtest code, smaller windows.
    """
    train_window, test_window = requested["train_window"], requested["test_window"]
    if n_available >= train_window + test_window:
        return {"train_window": train_window, "test_window": test_window}

    floor = MIN_WINDOW[horizon]
    scale = n_available / (train_window + test_window)
    new_train = max(floor["train_window"], int(train_window * scale * 0.9))
    new_test = max(floor["test_window"], int(test_window * scale * 0.9))

    if new_train + new_test > n_available:
        raise ValueError(
            f"Only {n_available} usable bars available after feature warm-up -- "
            f"too few for even a minimal walk-forward fold (need at least "
            f"{floor['train_window'] + floor['test_window']}). Fetch more "
            f"history, use a coarser timeframe, or backtest a different symbol."
        )
    print(
        f"Note: only {n_available} usable bars available -- shrinking walk-forward "
        f"windows from train={train_window}/test={test_window} to "
        f"train={new_train}/test={new_test} to fit."
    )
    return {"train_window": new_train, "test_window": new_test}


def walk_forward_predictions(
    df: pd.DataFrame,
    horizon: Horizon,
    lookahead: int,
    train_window: int,
    test_window: int,
    has_volume: bool = True,
) -> pd.DataFrame:
    """Builds features/labels from raw OHLCV, then runs the walk-forward
    core loop. Window sizes here are used as given -- callers wanting
    automatic shrinking on thin data should go through run_backtest_report.
    """
    feats = build_features(df, has_volume=has_volume)
    labels = build_labels(df, lookahead).reindex(feats.index)
    valid = labels.notna()
    feats, labels = feats[valid], labels[valid].astype(int)
    return _walk_forward_core(feats, labels, horizon, train_window, test_window)


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
    """One-call convenience wrapper used by main.py's --backtest flag.
    Resolves train/test window sizes against the data actually available
    (shrinking them if needed -- see resolve_windows) rather than assuming
    the configured defaults always fit.
    """
    feats = build_features(df, has_volume=has_volume)
    labels = build_labels(df, lookahead).reindex(feats.index)
    valid = labels.notna()
    feats, labels = feats[valid], labels[valid].astype(int)

    windows = resolve_windows(len(feats), horizon, {"train_window": train_window, "test_window": test_window})
    pooled = _walk_forward_core(feats, labels, horizon, windows["train_window"], windows["test_window"])

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
