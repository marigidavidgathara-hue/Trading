"""
features.py
Technical feature engineering shared by both horizons. Works on any OHLCV
DataFrame (crypto/forex/stock/futures, any bar size) coming out of
data_feed.py -- same feature set, the model layer is what differs per
horizon (different label lookahead, different training window).

All indicators are computed causally (only using data up to and including
the current bar) so nothing here leaks the future into training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta


REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def _validate(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {missing}")
    if df.isnull().any().any():
        raise ValueError("OHLCV frame contains NaNs -- clean before feature engineering")


def build_features(df: pd.DataFrame, has_volume: bool = True) -> pd.DataFrame:
    """Returns a new DataFrame of engineered features, indexed the same as
    `df`. Drops the warm-up rows that don't have enough history for the
    slowest indicator (longest lookback window).

    has_volume: set False for screen-capture feeds, where volume is not
    observable -- volume-based features are skipped rather than fed zeros.
    """
    _validate(df)
    out = pd.DataFrame(index=df.index)
    close, high, low, open_ = df["close"], df["high"], df["low"], df["open"]

    # --- returns / momentum ---------------------------------------------
    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_5"] = close.pct_change(5)
    out["log_ret_1"] = np.log(close).diff(1)
    out["hl_spread"] = (high - low) / close
    out["oc_spread"] = (close - open_) / open_

    # --- trend ------------------------------------------------------------
    out["sma_10"] = ta.trend.sma_indicator(close, window=10)
    out["sma_30"] = ta.trend.sma_indicator(close, window=30)
    out["sma_ratio_10_30"] = out["sma_10"] / out["sma_30"] - 1
    out["ema_12"] = ta.trend.ema_indicator(close, window=12)
    out["ema_26"] = ta.trend.ema_indicator(close, window=26)

    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_diff"] = macd.macd_diff()

    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    out["adx"] = adx.adx()

    # --- momentum oscillators ----------------------------------------------
    out["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    out["stoch_k"] = stoch.stoch()
    out["stoch_d"] = stoch.stoch_signal()
    out["roc_10"] = ta.momentum.ROCIndicator(close, window=10).roc()

    # --- volatility -------------------------------------------------------
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    out["bb_pct_b"] = bb.bollinger_pband()
    out["bb_width"] = bb.bollinger_wband()
    out["atr_14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    out["atr_pct"] = out["atr_14"] / close
    out["realized_vol_10"] = out["log_ret_1"].rolling(10).std()
    out["realized_vol_30"] = out["log_ret_1"].rolling(30).std()

    # --- volume -------------------------------------------------------------
    # Don't just trust the has_volume hint -- forex via yfinance reports
    # volume as exactly 0 on every bar (OTC market, no centralized tape), and
    # any ratio/pct_change against an all-zero series produces inf/NaN on
    # every row, which dropna() at the end would then turn into "no rows at
    # all". Require at least one genuinely nonzero reading before trusting it.
    volume_usable = (
        has_volume
        and "volume" in df
        and df["volume"].notna().any()
        and df["volume"].fillna(0).ne(0).any()
    )
    if volume_usable:
        vol = df["volume"]
        out["vol_chg_1"] = vol.pct_change(1)
        out["vol_sma_ratio"] = vol / vol.rolling(20).mean() - 1
        out["obv"] = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
        out["obv_chg"] = out["obv"].pct_change(5)
        out["mfi_14"] = ta.volume.MFIIndicator(high, low, close, vol, window=14).money_flow_index()

    # Any ratio/pct_change above can produce +/-inf when its denominator
    # hits exactly zero (e.g. a zero-volume bar in vol_chg_1, or OBV
    # crossing zero in obv_chg) -- this is a real occurrence with real
    # futures/forex data, not just an edge case. dropna() does NOT remove
    # inf, only NaN, so an infinite value would otherwise survive straight
    # into the matrix sklearn's scaler validates. An infinite ratio is
    # undefined, i.e. missing information for that row -- converting it to
    # NaN here makes dropna() treat it the same way it already treats
    # every other kind of missing feature value.
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna()


def build_labels(df: pd.DataFrame, lookahead: int) -> pd.Series:
    """Direction label for the close price `lookahead` bars ahead, in
    {0: down, 1: flat, 2: up}, using a small dead-zone around zero so tiny
    noise moves aren't forced into a direction. Aligned to the *current*
    bar's index (i.e. row t holds the label for the move from t to t+lookahead).
    """
    fwd_ret = df["close"].shift(-lookahead) / df["close"] - 1
    # dead zone scaled to the typical move size for this series, so it
    # behaves sensibly whether you're feeding 1m crypto bars or daily stock bars
    dead_zone = fwd_ret.abs().median() * 0.25
    labels = pd.Series(1, index=df.index)  # default: flat
    labels[fwd_ret > dead_zone] = 2   # up
    labels[fwd_ret < -dead_zone] = 0  # down
    labels.iloc[-lookahead:] = np.nan  # no forward data for the tail
    return labels


def build_dataset(df: pd.DataFrame, lookahead: int, has_volume: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Convenience wrapper: features + aligned label, NaNs dropped, both
    indexes matching. This is what model.py trains on."""
    feats = build_features(df, has_volume=has_volume)
    labels = build_labels(df, lookahead).reindex(feats.index)
    valid = labels.notna()
    return feats[valid], labels[valid].astype(int)
