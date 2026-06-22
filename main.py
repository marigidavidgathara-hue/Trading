"""
main.py
CLI entry point wiring everything together.

    # Crypto, scalp horizon, API feed: validate the approach on history first
    python main.py backtest --market crypto --symbol BTC/USDT --horizon scalp

    # Train a model on recent history and save it
    python main.py train --market crypto --symbol BTC/USDT --horizon scalp

    # One-shot prediction using the saved model + a fresh API pull
    python main.py predict --market crypto --symbol BTC/USDT --horizon scalp

    # Continuous predictions, logged to predictions_log.csv, using the
    # screen feed instead of an API (run screen_vision.py --calibrate first)
    python main.py live --market crypto --symbol BTC/USDT --horizon scalp --source screen

Always backtest before trusting `predict`/`live` output -- an
untested model is just a number generator.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import pandas as pd

from config import (
    MarketType, Horizon,
    get_timeframe, HORIZON_LABEL_LOOKAHEAD,
    DEFAULT_SYMBOLS, PREDICTIONS_LOG_PATH,
)
from data_feed import get_feed
from features import build_features, build_dataset
from model import DirectionModel
from backtest import run_backtest_report

# Walk-forward window sizes per horizon, in bars. Scalp uses 1m bars so the
# windows are wide in bar-count but still a few days of wall time; swing
# uses daily bars so windows span roughly a trading year.
WALK_FORWARD_WINDOWS = {
    Horizon.SCALP: dict(train_window=1500, test_window=300),
    Horizon.SWING: dict(train_window=250, test_window=50),
}

HISTORY_LIMIT = {
    Horizon.SCALP: 5000,
    Horizon.SWING: 1500,
}


def _fetch(market: MarketType, symbol: str, horizon: Horizon, source: str) -> pd.DataFrame:
    feed = get_feed(market, source)
    timeframe = get_timeframe(market, horizon)
    return feed.get_ohlcv(symbol, timeframe, limit=HISTORY_LIMIT[horizon])


def cmd_backtest(args):
    df = _fetch(args.market, args.symbol, args.horizon, args.source)
    lookahead = HORIZON_LABEL_LOOKAHEAD[args.horizon]
    windows = WALK_FORWARD_WINDOWS[args.horizon]
    has_volume = args.source == "api"  # screen feed never has volume

    report = run_backtest_report(
        df, args.horizon, lookahead,
        train_window=windows["train_window"], test_window=windows["test_window"],
        has_volume=has_volume,
        confidence_threshold=args.confidence_threshold,
        cost_per_trade=args.cost_per_trade,
    )
    print(f"\n=== Backtest: {args.symbol} ({args.horizon.value}, {args.source}) ===")
    for k, v in report["scores"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nCalibration table:\n{report['calibration_table']}")
    print(f"\nTrades taken: {report['n_trades']}  |  Win rate: {report['win_rate_on_traded']:.3f}"
          f"  |  Final equity (per $1, after {args.cost_per_trade*1e4:.1f}bps/trade): {report['final_equity']:.4f}")
    print(
        "\nReminder: this is in-sample-of-history performance with a simple cost "
        "model. It does not guarantee live results. Not financial advice."
    )
    return report


def cmd_train(args):
    df = _fetch(args.market, args.symbol, args.horizon, args.source)
    lookahead = HORIZON_LABEL_LOOKAHEAD[args.horizon]
    has_volume = args.source == "api"
    X, y = build_dataset(df, lookahead, has_volume=has_volume)

    model = DirectionModel(args.horizon)
    model.fit(X, y)
    metrics = model.evaluate(X, y)
    path = model.save(args.symbol)
    print(f"Trained on {len(X)} bars. In-sample sanity metrics: {metrics}")
    print(f"Saved model to {path}")
    print("Run `backtest` (out-of-sample, walk-forward) before trusting this for live predictions.")
    return model, metrics, path


def cmd_predict(args):
    feed = get_feed(args.market, args.source)
    timeframe = get_timeframe(args.market, args.horizon)
    has_volume = args.source == "api"

    df = feed.get_ohlcv(args.symbol, timeframe, limit=200)
    feats = build_features(df, has_volume=has_volume)
    model = DirectionModel.load(args.symbol, args.horizon)
    result = model.predict_direction(feats.tail(1))
    probs = model.predict_proba(feats.tail(1))
    last_price = df["close"].iloc[-1]

    print(f"{args.symbol} @ {last_price:.4f} -> {result['direction'].iloc[-1]} "
          f"(confidence {result['confidence'].iloc[-1]:.3f})")
    print(probs.iloc[-1].to_dict())
    return feats.tail(1), result, probs, last_price


def _log_prediction(symbol, horizon, source, price, direction, confidence, probs_row, feats_row):
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "horizon": horizon.value,
        "source": source,
        "price": price,
        "direction": direction,
        "confidence": confidence,
        "p_down": probs_row["p_down"],
        "p_flat": probs_row["p_flat"],
        "p_up": probs_row["p_up"],
        # raw feature snapshot, same idea as the football system's
        # feature_json column -- lets you re-score/recalibrate later
        # without recomputing features from scratch.
        "feature_json": json.dumps(feats_row.to_dict()),
    }
    header_needed = not PREDICTIONS_LOG_PATH.exists()
    pd.DataFrame([row]).to_csv(PREDICTIONS_LOG_PATH, mode="a", header=header_needed, index=False)


def cmd_live(args):
    model = DirectionModel.load(args.symbol, args.horizon)
    feed = get_feed(args.market, args.source)
    timeframe = get_timeframe(args.market, args.horizon)
    has_volume = args.source == "api"

    print(f"Live predictions for {args.symbol} every {args.poll_seconds}s. Ctrl+C to stop.")
    print(f"Logging to {PREDICTIONS_LOG_PATH}")
    try:
        while True:
            df = feed.get_ohlcv(args.symbol, timeframe, limit=200)
            feats = build_features(df, has_volume=has_volume)
            if feats.empty:
                print("Not enough bars yet for a feature window...")
            else:
                row = feats.iloc[-1:]
                probs = model.predict_proba(row).iloc[-1]
                pred = model.predict_direction(row).iloc[-1]
                price = df["close"].iloc[-1]
                ts = df.index[-1]
                print(f"[{ts}] {args.symbol} @ {price:.4f} -> {pred['direction']} "
                      f"(confidence {pred['confidence']:.3f})")
                _log_prediction(args.symbol, args.horizon, args.source, price,
                                 pred["direction"], pred["confidence"], probs, row.iloc[-1])
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("Stopped.")


def build_parser():
    p = argparse.ArgumentParser(description="Trading direction prediction system")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--market", type=str, choices=[m.value for m in MarketType], required=True)
        sp.add_argument("--symbol", type=str, default=None, help="Defaults to config.DEFAULT_SYMBOLS[market]")
        sp.add_argument("--horizon", type=str, choices=[h.value for h in Horizon], required=True)
        sp.add_argument("--source", type=str, choices=["api", "screen"], default="api")

    bt = sub.add_parser("backtest")
    add_common(bt)
    bt.add_argument("--confidence-threshold", dest="confidence_threshold", type=float, default=0.5)
    bt.add_argument("--cost-per-trade", dest="cost_per_trade", type=float, default=0.0005)
    bt.set_defaults(func=cmd_backtest)

    tr = sub.add_parser("train")
    add_common(tr)
    tr.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict")
    add_common(pr)
    pr.set_defaults(func=cmd_predict)

    lv = sub.add_parser("live")
    add_common(lv)
    lv.add_argument("--poll-seconds", dest="poll_seconds", type=float, default=5.0)
    lv.set_defaults(func=cmd_live)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.market = MarketType(args.market)
    args.horizon = Horizon(args.horizon)
    if args.symbol is None:
        args.symbol = DEFAULT_SYMBOLS[args.market]
    args.func(args)


if __name__ == "__main__":
    main()
