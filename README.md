# Trading Direction Prediction System

A modular multi-asset, multi-horizon direction-prediction engine, structured
the same way as the football prediction codebase: a swappable data layer,
shared feature engineering, a calibrated ensemble model, and a walk-forward
backtest harness you run *before* trusting any live output.

**Not financial advice.** This predicts a probability distribution over
{down, flat, up}; it does not guarantee profit, and markets can and do stop
behaving like their recent history. Treat every number this system produces
as a probabilistic opinion to weigh, not an instruction.

## Architecture

```
config.py        Symbols, timeframes, horizons, screen calibration storage
data_feed.py      DataFeed interface; APIDataFeed (ccxt/yfinance), ScreenCaptureDataFeed
screen_vision.py  Screen calibration tool + OCR extraction for the screen feed
features.py       Technical indicators (trend/momentum/volatility/volume) + labels
model.py          Per-(symbol, horizon) calibrated ensemble (LightGBM + LogisticRegression)
backtest.py        Walk-forward validation: Brier score, calibration table, equity curve
main.py            CLI: backtest / train / predict / live
```

`features.py`, `model.py`, and `backtest.py` only ever talk to the
`DataFeed` interface, so the same model code runs unmodified whether the
bars came from an exchange API or from reading your screen.

## Markets supported

| Market   | Backend (API mode) | Example symbol |
|----------|--------------------|----------------|
| Crypto   | `ccxt` (Binance, public data, no key needed) | `BTC/USDT` |
| Forex    | `yfinance`         | `EURUSD=X` |
| Stocks   | `yfinance`         | `AAPL` |
| Futures  | `yfinance`         | `ES=F` |

Edit `DEFAULT_SYMBOLS` / `CCXT_EXCHANGE` in `config.py` for your own
symbols or exchange. Swap `ccxt`'s exchange class for any other ccxt-supported
exchange (Kraken, Coinbase, etc.) if you trade somewhere other than Binance.

## Horizons

- **Scalp** (`scalp`): 1-minute bars, predicts direction 3 bars ahead.
- **Swing** (`swing`): daily bars, predicts next day's direction.

Both share the same feature/model code (`HORIZON_TIMEFRAMES` and
`HORIZON_LABEL_LOOKAHEAD` in `config.py` is all that differs). Add an
intraday horizon the same way if you want one later.

## Data sources

- **`api`** (default, recommended): pulls real OHLCV bars directly from the
  exchange/yfinance. Reliable, has real volume, no calibration step.
- **`screen`**: reads a price label (and optional indicator readouts) off
  your actual screen via screenshot + OCR. Use this only if you need
  something that's *not* available through an API — a proprietary
  indicator on your platform, a broker with no API access, etc. It has no
  access to historical bars (a screenshot only shows "now"), no volume,
  and OCR will occasionally misread digits. It builds up bars over time by
  resampling polled price ticks, so give it a while running before
  expecting a useful feature window.

### Setting up the screen feed

```bash
python screen_vision.py --calibrate
```
This takes one screenshot, then lets you drag boxes over the price label,
the chart area, and any indicator readouts you want OCR'd. Saves to
`screen_calibration.json`. Re-run this any time you move the window,
change zoom level, or switch monitors. Sanity-check it with:
```bash
python screen_vision.py --test
```

## Usage

```bash
pip install -r requirements.txt

# 1. Always backtest first -- walk-forward, out-of-sample, with a cost model
python main.py backtest --market crypto --symbol BTC/USDT --horizon scalp

# 2. If the backtest looks reasonable, train the model you'll actually use
python main.py train --market crypto --symbol BTC/USDT --horizon scalp

# 3. One-shot prediction
python main.py predict --market crypto --symbol BTC/USDT --horizon scalp

# 4. Continuous predictions, logged to predictions_log.csv
python main.py live --market crypto --symbol BTC/USDT --horizon scalp --poll-seconds 5

# Screen feed instead of API (after calibrating):
python main.py live --market crypto --symbol BTC/USDT --horizon scalp --source screen
```

`predictions_log.csv` stores every live prediction plus a JSON snapshot of
the feature row that produced it — same pattern as the football system's
`feature_json` column — so you can recalibrate or audit later without
recomputing features from scratch.

## What the backtest actually tells you

`run_backtest_report()` slides a training window forward in fixed steps,
refits each time, and only ever predicts on the untouched slice immediately
after — no shuffling, no peeking forward. It reports:

- **Accuracy / log loss / Brier score** on pooled out-of-sample predictions.
- **Calibration table**: bins predictions by the model's own stated
  confidence and checks whether it was actually right that often in each
  bin. A model that says "70% confident" should be right ~70% of the time
  in that bin — if it isn't, the probabilities aren't trustworthy yet.
- **Equity curve**: a naive simulation that takes every prediction above a
  confidence threshold, holds for the label's lookahead, and nets out a
  flat per-trade cost (`--cost-per-trade`, in fraction terms — 0.0005 = 5
  bps round-trip). This ignores slippage beyond that flat fee, position
  overlap, and sizing — it's a sanity check on whether there's any edge
  left after costs, not an execution model.

If the backtest results don't hold up, don't skip to `train`/`live` anyway
hoping it'll be different this time — that's the model telling you it
hasn't found anything real in this symbol/horizon yet.

## Extending

- **New horizon**: add an entry to `Horizon`, `HORIZON_TIMEFRAMES`,
  `HORIZON_LABEL_LOOKAHEAD` in `config.py`, and a window size in
  `WALK_FORWARD_WINDOWS` in `main.py`.
- **New market/exchange**: add a `MarketType`, branch in `APIDataFeed`.
- **Better blend**: `model.py`'s blend is a flat average of LightGBM +
  LogisticRegression probabilities by design — simple and hard to overfit
  on a small calibration slice. If you later have enough out-of-fold data
  to justify it, replace `_blend_raw`'s average with a learned meta-weight
  (same lesson as the football system's MetaLearner: don't add learned
  blending weights until you have the data to support them, or you'll
  reproduce the same Brier-score regression that forced the `C=0.05` fix
  there).
- **Chart-pixel candle reconstruction**: `screen_vision.py` currently only
  OCRs numeric labels (price, indicators). Reconstructing OHLC bars
  directly from candle pixel colors is possible with OpenCV color masking
  but is a meaningfully bigger, more fragile lift — only worth building if
  the polled-price-tick resampling in `ScreenCaptureDataFeed` proves too
  coarse for your use case.
