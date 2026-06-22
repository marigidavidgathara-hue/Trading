"""
data_feed.py
Unified OHLCV interface with two interchangeable backends:

  APIDataFeed          -- ccxt (crypto) / yfinance (forex, stocks, futures)
  ScreenCaptureDataFeed -- mss screenshot -> screen_vision extraction

Both return the same shape: a DataFrame indexed by UTC timestamp with
columns [open, high, low, close, volume]. Everything downstream
(features.py, model.py, backtest.py) only ever talks to this interface,
so swapping API <-> screen requires no changes elsewhere.
"""

from __future__ import annotations

import abc
import logging
import time
from datetime import datetime, timezone

import pandas as pd

from config import MarketType, CCXT_EXCHANGE, ScreenConfig

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Our timeframe strings follow ccxt/yfinance convention ("1m", "5m", "1h",
# "1d"). pandas resample() uses different aliases -- bare "m" means
# month-end there, not minute -- so this maps the small set we use to the
# warning-free pandas equivalents. Only ScreenCaptureDataFeed needs this;
# ccxt/yfinance take the original strings directly.
_PANDAS_FREQ = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "4h": "4h", "1d": "1D",
}


def _to_pandas_freq(timeframe: str) -> str:
    if timeframe not in _PANDAS_FREQ:
        raise ValueError(
            f"Unsupported timeframe for the screen-capture feed's resampling: {timeframe!r}. "
            f"Supported: {list(_PANDAS_FREQ)}"
        )
    return _PANDAS_FREQ[timeframe]


class DataFeed(abc.ABC):
    """Common interface every data source backend implements."""

    @abc.abstractmethod
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """Return historical/recent OHLCV bars, oldest first, UTC-indexed."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """Return the most recent traded/quoted price."""
        raise NotImplementedError


# --------------------------------------------------------------------------
# API-backed feed
# --------------------------------------------------------------------------

class APIDataFeed(DataFeed):
    """Routes to ccxt for crypto and yfinance for forex/stocks/futures,
    behind one interface. No screen involved -- this is the reliable path.
    """

    def __init__(self, market_type: MarketType):
        self.market_type = MarketType(market_type)
        self._ccxt_client = None  # lazy init, only needed for crypto

    # -- crypto via ccxt --------------------------------------------------
    def _ccxt(self):
        if self._ccxt_client is None:
            import ccxt
            exchange_cls = getattr(ccxt, CCXT_EXCHANGE)
            self._ccxt_client = exchange_cls({"enableRateLimit": True})
        return self._ccxt_client

    def _get_ohlcv_crypto(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        client = self._ccxt()
        raw = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts")
        return df[OHLCV_COLUMNS]

    def _get_latest_price_crypto(self, symbol: str) -> float:
        ticker = self._ccxt().fetch_ticker(symbol)
        return float(ticker["last"])

    # -- forex / stocks / futures via yfinance ----------------------------
    @staticmethod
    def _yf_interval(timeframe: str) -> str:
        # yfinance uses its own interval vocabulary; map the small set we use.
        mapping = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "1d": "1d"}
        if timeframe not in mapping:
            raise ValueError(f"Unsupported timeframe for yfinance backend: {timeframe}")
        return mapping[timeframe]

    @staticmethod
    def _yf_period_for(interval: str, limit: int) -> str:
        # yfinance limits how far back intraday intervals can go; pick a
        # period comfortably covering `limit` bars without over-requesting.
        if interval.endswith("m"):
            return "7d" if interval == "1m" else "60d"
        return f"{max(limit * 2, 60)}d"

    def _get_ohlcv_yf(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        import yfinance as yf
        interval = self._yf_interval(timeframe)
        period = self._yf_period_for(interval, limit)

        # Yahoo's free endpoint is well-documented to rate-limit rapid
        # repeated requests, and surfaces that as a generic "possibly
        # delisted; no price data found" rather than a clear rate-limit
        # error -- even for a symbol that just worked seconds earlier
        # (e.g. backtest succeeding, then train failing on the identical
        # call right after). Retrying with backoff is the standard fix for
        # this specific, well-known failure mode, not a real data problem.
        last_error = None
        df = None
        max_attempts = 4
        for attempt in range(max_attempts):
            if attempt > 0:
                time.sleep(3 * attempt)  # 3s, 6s, 9s backoff between retries
            try:
                df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
            except Exception as e:
                last_error = e
                df = None
                continue
            if not df.empty:
                break
            last_error = RuntimeError(f"yfinance returned no data for {symbol} ({interval}, {period})")
            df = None

        if df is None or df.empty:
            raise RuntimeError(
                f"yfinance returned no data for {symbol} ({interval}, {period}) "
                f"after {max_attempts} attempts with backoff. This is usually "
                f"Yahoo Finance rate-limiting the free endpoint rather than a bad "
                f"symbol -- it often succeeds again after waiting a bit longer."
            ) from last_error
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)[OHLCV_COLUMNS]
        df.index = pd.to_datetime(df.index, utc=True)
        return df.tail(limit)

    def _get_latest_price_yf(self, symbol: str) -> float:
        df = self._get_ohlcv_yf(symbol, "1m", 1)
        return float(df["close"].iloc[-1])

    # -- public interface ---------------------------------------------------
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        if self.market_type == MarketType.CRYPTO:
            return self._get_ohlcv_crypto(symbol, timeframe, limit)
        return self._get_ohlcv_yf(symbol, timeframe, limit)

    def get_latest_price(self, symbol: str) -> float:
        if self.market_type == MarketType.CRYPTO:
            return self._get_latest_price_crypto(symbol)
        return self._get_latest_price_yf(symbol)


# --------------------------------------------------------------------------
# Screen-capture-backed feed
# --------------------------------------------------------------------------

class ScreenCaptureDataFeed(DataFeed):
    """Reads price/chart data directly off your screen instead of an API.

    Use this when you need to mirror exactly what's rendered on a specific
    platform (custom indicators, a broker with no API, etc). It is
    fundamentally less reliable than APIDataFeed: OCR misreads digits,
    chart-to-candle reconstruction from pixels is approximate, and it only
    works while the calibrated window is visible, on the same monitor
    layout, at the same zoom level used during calibration.

    Requires running `python screen_vision.py --calibrate` once on the
    machine where the trading platform is actually open.
    """

    def __init__(self, screen_config: ScreenConfig | None = None):
        self.cfg = screen_config or ScreenConfig.load()
        self._history: list[dict] = []  # rolling buffer of polled bars, since
        # the screen only ever gives you "now", not history

    def poll_once(self) -> dict:
        """Grab one frame, extract price (+ indicators if calibrated), and
        append it to the in-memory rolling buffer as a synthetic 1-bar
        sample. Call this on a timer (see main.py) to build up a series."""
        from screen_vision import extract_price, extract_indicators

        price = extract_price(self.cfg)
        indicators = extract_indicators(self.cfg) if self.cfg.indicator_regions else {}
        sample = {"ts": datetime.now(timezone.utc), "price": price, **indicators}
        self._history.append(sample)
        return sample

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """Builds OHLCV bars by resampling the polled price ticks. Note this
        only has data from the moment polling started -- there's no
        history to pull retroactively from a screenshot. Run main.py's
        live loop for a while before expecting a usable window of bars.
        """
        if not self._history:
            raise RuntimeError(
                "No screen samples collected yet -- call poll_once() repeatedly "
                "(e.g. via the live loop in main.py) before requesting OHLCV bars."
            )
        df = pd.DataFrame(self._history).set_index("ts")
        ohlc = df["price"].resample(_to_pandas_freq(timeframe)).ohlc()
        ohlc["volume"] = float("nan")  # not observable from a screenshot
        return ohlc.dropna(subset=["open"]).tail(limit)

    def get_latest_price(self, symbol: str) -> float:
        sample = self.poll_once()
        return sample["price"]


def get_feed(market_type: MarketType, mode: str) -> DataFeed:
    """Factory: mode is 'api' or 'screen'. Same call site, swappable backend."""
    if mode == "api":
        return APIDataFeed(market_type)
    if mode == "screen":
        return ScreenCaptureDataFeed()
    raise ValueError(f"Unknown data source mode: {mode}")
