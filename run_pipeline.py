"""
run_pipeline.py
Runs the full sequence in one process, in the order it actually needs to
happen -- you should never train/predict on a symbol you haven't backtested:

    1. backtest  -- walk-forward, out-of-sample. If this doesn't look right,
                     everything downstream is just confidently-wrong numbers.
    2. train     -- fit + save the production model on full available history
    3. predict   -- one-shot prediction with the model you just trained
    4. live      -- optional continuous polling loop (off unless --live)

This doesn't reimplement anything -- it calls main.py's existing
cmd_backtest / cmd_train / cmd_predict / cmd_live functions in sequence, so
behavior is identical to running each `python main.py <cmd> ...` by hand.

Usage:
    python run_pipeline.py --market crypto --symbol BTC/USDT --horizon scalp
    python run_pipeline.py --market stock --symbol AAPL --horizon swing --live
    python run_pipeline.py --market crypto --horizon scalp --min-accuracy 0.4
"""

from __future__ import annotations

import argparse
import sys

import main as cli
from config import MarketType, Horizon, DEFAULT_SYMBOLS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run backtest -> train -> predict -> (optional) live, in order")
    p.add_argument("--market", type=str, choices=[m.value for m in MarketType], required=True)
    p.add_argument("--symbol", type=str, default=None, help="Defaults to config.DEFAULT_SYMBOLS[market]")
    p.add_argument("--horizon", type=str, choices=[h.value for h in Horizon], required=True)
    p.add_argument("--source", type=str, choices=["api", "screen"], default="api")
    p.add_argument("--confidence-threshold", dest="confidence_threshold", type=float, default=0.5)
    p.add_argument("--cost-per-trade", dest="cost_per_trade", type=float, default=0.0005)
    p.add_argument("--skip-backtest", action="store_true",
                    help="Skip straight to train/predict (not recommended)")
    p.add_argument("--min-accuracy", type=float, default=None,
                    help="If set, abort before training when backtest accuracy is below this")
    p.add_argument("--live", action="store_true",
                    help="After the one-shot prediction, keep polling continuously (Ctrl+C to stop)")
    p.add_argument("--poll-seconds", dest="poll_seconds", type=float, default=5.0)
    return p


def _banner(step: str, total: int, label: str):
    print(f"\n{'='*70}\nSTEP {step}/{total}: {label}\n{'='*70}")


def run(args: argparse.Namespace):
    if args.symbol is None:
        args.symbol = DEFAULT_SYMBOLS[args.market]
    total = 4 if args.live else 3

    # 1. backtest -- gate everything downstream on this
    _banner("1", total, f"BACKTEST  ({args.symbol}, {args.horizon.value}, source={args.source})")
    if args.skip_backtest:
        print("Skipped (--skip-backtest passed).")
    else:
        report = cli.cmd_backtest(args)
        if args.min_accuracy is not None and report["scores"]["accuracy"] < args.min_accuracy:
            print(
                f"\nAborting: backtest accuracy {report['scores']['accuracy']:.3f} "
                f"is below --min-accuracy {args.min_accuracy:.3f}. "
                f"Not training/predicting on this symbol/horizon as-is."
            )
            sys.exit(1)

    # 2. train -- fit + save the model that predict/live will load
    _banner("2", total, f"TRAIN  ({args.symbol}, {args.horizon.value})")
    cli.cmd_train(args)

    # 3. predict -- one-shot, using the model just saved above
    _banner("3", total, f"PREDICT  ({args.symbol}, {args.horizon.value})")
    cli.cmd_predict(args)

    # 4. live -- optional, runs until Ctrl+C
    if args.live:
        _banner("4", total, f"LIVE  ({args.symbol}, {args.horizon.value}, every {args.poll_seconds}s)")
        cli.cmd_live(args)

    print(f"\n{'='*70}\nPipeline complete.\n{'='*70}")


def main():
    args = build_parser().parse_args()
    args.market = MarketType(args.market)
    args.horizon = Horizon(args.horizon)
    run(args)


if __name__ == "__main__":
    main()