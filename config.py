"""
config.py
Central configuration for the trading prediction system.
Edit this file to point at your own symbols / screen regions / model paths.
No logic lives here -- only settings.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class MarketType(str, Enum):
    CRYPTO = "crypto"
    FOREX = "forex"
    STOCK = "stock"
    FUTURES = "futures"


class Horizon(str, Enum):
    SCALP = "scalp"   # seconds-minutes bars, predicting next few bars
    SWING = "swing"   # daily bars, predicting next day/week


class DataSourceMode(str, Enum):
    API = "api"
    SCREEN = "screen"


# Bar size used to build features/labels for each horizon.
# ccxt/yfinance timeframe strings.
HORIZON_TIMEFRAMES = {
    Horizon.SCALP: "1m",
    Horizon.SWING: "1d",
}

# How many bars ahead the label looks when building the target
# (e.g. scalp -> direction of close 3 bars from now; swing -> next day's close)
HORIZON_LABEL_LOOKAHEAD = {
    Horizon.SCALP: 3,
    Horizon.SWING: 1,
}

# Default symbols per market, used by examples / CLI. Replace with your own.
DEFAULT_SYMBOLS = {
    MarketType.CRYPTO: "BTC/USDT",       # ccxt unified symbol (Binance)
    MarketType.FOREX: "EURUSD=X",        # yfinance ticker
    MarketType.STOCK: "AAPL",            # yfinance ticker
    MarketType.FUTURES: "ES=F",          # yfinance ticker (S&P 500 e-mini)
}

# Which exchange ccxt should hit for crypto. Public market data, no key needed.
CCXT_EXCHANGE = "binance"


@dataclass
class ScreenRegion:
    """Pixel rectangle on the monitor, top-left origin. Produced by the
    calibration tool in screen_vision.py -- don't hand-guess these."""
    name: str
    left: int
    top: int
    width: int
    height: int

    def as_mss_dict(self) -> dict:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


@dataclass
class ScreenConfig:
    """Calibration for the ScreenCaptureDataFeed. Populate via
    `python screen_vision.py --calibrate` which writes this out as JSON;
    load it back with ScreenConfig.load().
    """
    price_region: ScreenRegion = None          # crop showing the last-traded price label
    chart_region: ScreenRegion = None           # crop of the candlestick chart area
    indicator_regions: dict = field(default_factory=dict)  # name -> ScreenRegion, for custom on-screen indicators
    monitor_index: int = 1                      # mss monitor index (1 = primary)
    poll_seconds: float = 1.0                   # how often to grab a frame

    @staticmethod
    def calibration_path() -> Path:
        return Path(__file__).parent / "screen_calibration.json"

    @classmethod
    def load(cls):
        import json
        p = cls.calibration_path()
        if not p.exists():
            raise FileNotFoundError(
                f"No screen calibration found at {p}. Run "
                f"`python screen_vision.py --calibrate` first."
            )
        data = json.loads(p.read_text())
        cfg = cls()
        cfg.price_region = ScreenRegion(**data["price_region"]) if data.get("price_region") else None
        cfg.chart_region = ScreenRegion(**data["chart_region"]) if data.get("chart_region") else None
        cfg.indicator_regions = {
            k: ScreenRegion(**v) for k, v in data.get("indicator_regions", {}).items()
        }
        cfg.monitor_index = data.get("monitor_index", 1)
        cfg.poll_seconds = data.get("poll_seconds", 1.0)
        return cfg

    def save(self):
        import json, dataclasses
        data = {
            "price_region": dataclasses.asdict(self.price_region) if self.price_region else None,
            "chart_region": dataclasses.asdict(self.chart_region) if self.chart_region else None,
            "indicator_regions": {k: dataclasses.asdict(v) for k, v in self.indicator_regions.items()},
            "monitor_index": self.monitor_index,
            "poll_seconds": self.poll_seconds,
        }
        self.calibration_path().write_text(json.dumps(data, indent=2))


# Where trained per-horizon models get pickled to.
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Predictions log (mirrors the football system's predictions log -- needed
# to ever backtest/calibrate against your own live track record).
PREDICTIONS_LOG_PATH = Path(__file__).parent / "predictions_log.csv"
