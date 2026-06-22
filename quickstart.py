"""
quickstart.py
The simplest way to run this: answer two or three plain questions, no
flags, no "horizon" jargon. Just:

    python quickstart.py

Then type things like "forex" and "5m" when asked.

Two settings people often mix up:
  - Candle size (1m, 5m, 1h...) -- the size of each bar the model reads.
  - Refresh rate -- how often it checks for a fresh prediction.
These are independent. Sub-minute candles ("5 seconds", "5s") aren't
available from any free data source this uses -- Yahoo Finance's floor is
1-minute bars, and that's the practical floor for crypto exchange history
too. But the refresh rate absolutely can be 5 seconds regardless of candle
size -- that's the second question below, not the first.
"""

from __future__ import annotations

import sys

import config as cfg
from config import MarketType, Horizon, DEFAULT_SYMBOLS
import run_pipeline

# What's actually reliable per market given free-API limits. Crypto (via
# ccxt) has deep continuous history even at 1-minute resolution. Forex/
# stocks/futures (via yfinance) have no sub-minute bars at all and only
# keep 1-minute history for 7 days, so 5-minute-and-up is the dependable
# range -- see config.py's HORIZON_TIMEFRAMES comment for the full reasoning.
RECOMMENDED_TIMEFRAMES = {
    MarketType.CRYPTO: ["1m", "5m", "15m", "1h", "1d"],
    MarketType.FOREX: ["5m", "15m", "1h", "1d"],
    MarketType.STOCK: ["5m", "15m", "1h", "1d"],
    MarketType.FUTURES: ["5m", "15m", "1h", "1d"],
}

SUB_MINUTE_HINTS = ("1s", "5s", "10s", "15s", "30s", "second")


def _ask(prompt: str) -> str:
    return input(prompt).strip().lower()


def _ask_preserve_case(prompt: str) -> str:
    """Same as _ask but doesn't lowercase -- tickers are case-sensitive
    (EURUSD=X, BTC/USDT, AAPL), unlike the market/timeframe answers."""
    return input(prompt).strip()


def main():
    print("Quickstart -- a few plain questions, then it runs.\n")

    market_raw = _ask("Market? (crypto / forex / stock / futures): ")
    try:
        market = MarketType(market_raw)
    except ValueError:
        print(f"Didn't recognize '{market_raw}'. Pick one of: crypto, forex, stock, futures.")
        sys.exit(1)

    options = RECOMMENDED_TIMEFRAMES[market]
    print(f"\nTimeframe options for {market.value}: {', '.join(options)}")
    tf_raw = _ask("Timeframe? (e.g. 5m): ")

    if any(hint in tf_raw for hint in SUB_MINUTE_HINTS):
        timeframe = options[0]
        print(
            "\nSub-minute candles aren't available from the free data sources "
            "this uses -- Yahoo Finance's floor is 1-minute bars, and that's "
            f"also the practical floor for crypto history. Using {timeframe} "
            "candles instead.\n"
            "(Want predictions to refresh every 5 seconds regardless of candle "
            "size? That's the next question, not this one.)"
        )
    elif tf_raw in options:
        timeframe = tf_raw
    else:
        timeframe = options[0]
        print(f"\n'{tf_raw}' isn't one of the supported options for {market.value} -- using {timeframe} instead.")

    poll_raw = _ask("\nHow often should it check for a fresh prediction, in seconds? [5]: ") or "5"
    try:
        poll_seconds = float(poll_raw)
    except ValueError:
        print(f"Didn't understand '{poll_raw}' -- using 5 seconds.")
        poll_seconds = 5.0

    symbol = _ask_preserve_case(f"\nSymbol? [{DEFAULT_SYMBOLS[market]}]: ") or DEFAULT_SYMBOLS[market]

    # Daily-or-slower candles reuse the "swing" lookahead/backtest-window
    # defaults; anything finer reuses "scalp"'s. The chosen literal
    # timeframe then overrides the bar size those defaults normally imply.
    horizon = Horizon.SWING if timeframe.endswith(("d", "w")) else Horizon.SCALP
    cfg.HORIZON_TIMEFRAMES[market][horizon] = timeframe

    live_raw = _ask("\nKeep running live after the first prediction? (y/n) [n]: ")
    go_live = live_raw.startswith("y")

    print(
        f"\nRunning: market={market.value}  symbol={symbol}  timeframe={timeframe}  "
        f"refresh={poll_seconds}s  live={go_live}\n"
    )

    args = run_pipeline.build_parser().parse_args([
        "--market", market.value,
        "--symbol", symbol,
        "--horizon", horizon.value,
        "--poll-seconds", str(poll_seconds),
    ])
    args.market = market
    args.horizon = horizon
    args.live = go_live
    run_pipeline.run(args)


if __name__ == "__main__":
    main()
