"""
screen_vision.py
Screen capture + extraction for ScreenCaptureDataFeed.

Two things live here:
  1. A one-time interactive calibration tool (`--calibrate`) that lets you
     draw boxes over your trading platform to mark where the price label
     and any indicators you care about are rendered.
  2. Extraction functions that re-capture those same regions on demand and
     OCR the numbers out of them.

This module needs an actual display and an actual trading platform window
on screen -- it will not do anything useful in a headless environment.
Run it on the machine where your platform is open.

OCR honesty note: this works best on plain, high-contrast, fixed-position
numeric labels (a price ticker, a single indicator readout). It is not a
substitute for an API feed for anything that requires precision -- treat
ScreenCaptureDataFeed output as approximate.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from config import ScreenConfig, ScreenRegion

logger = logging.getLogger(__name__)

_NUMERIC_OCR_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789.,-"


def capture_region(region: ScreenRegion) -> np.ndarray:
    """Grab one region of the screen as a BGR numpy array."""
    import mss
    import cv2

    with mss.mss() as sct:
        shot = sct.grab(region.as_mss_dict())
        img = np.array(shot)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def _preprocess_for_ocr(img_bgr: np.ndarray, scale: int = 3):
    """Upscale + adaptive-threshold a small UI crop so tesseract has a
    fighting chance against thin anti-aliased platform fonts."""
    import cv2
    from PIL import Image

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    return Image.fromarray(thresh)


def _ocr_numeric(img_bgr: np.ndarray) -> str:
    import pytesseract

    pil_img = _preprocess_for_ocr(img_bgr)
    return pytesseract.image_to_string(pil_img, config=_NUMERIC_OCR_CONFIG)


def _parse_number(raw_text: str) -> float:
    cleaned = re.sub(r"[^0-9.\-]", "", raw_text.replace(",", ""))
    if not cleaned or cleaned in {"-", "."}:
        raise ValueError(f"OCR produced unparseable text: {raw_text!r}")
    return float(cleaned)


def extract_price(cfg: ScreenConfig) -> float:
    if cfg.price_region is None:
        raise ValueError("No price_region calibrated. Run: python screen_vision.py --calibrate")
    img = capture_region(cfg.price_region)
    raw = _ocr_numeric(img)
    return _parse_number(raw)


def extract_indicators(cfg: ScreenConfig) -> dict:
    """OCR every calibrated indicator region. Returns None for any region
    that failed to parse instead of raising, so one bad indicator doesn't
    take down the whole poll."""
    out = {}
    for name, region in cfg.indicator_regions.items():
        img = capture_region(region)
        raw = _ocr_numeric(img)
        try:
            out[name] = _parse_number(raw)
        except ValueError:
            logger.warning("Indicator %r: could not parse OCR text %r", name, raw)
            out[name] = None
    return out


def calibrate():
    """Interactive setup: screenshot the primary monitor, then drag boxes
    over the price label, the chart area, and any indicator readouts.
    Requires a real display -- run this locally, not in a sandbox/CI.
    """
    import cv2
    import mss

    print(
        "Make sure your trading platform is visible in its normal position.\n"
        "A screenshot will be taken when you press Enter."
    )
    input("Press Enter to capture... ")

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        full = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

    cfg = ScreenConfig()
    cfg.monitor_index = 1

    def select(label: str) -> ScreenRegion:
        win = f"Drag a box: {label} (Enter/Space to confirm, Esc to skip)"
        x, y, w, h = cv2.selectROI(win, full, showCrosshair=True)
        cv2.destroyWindow(win)
        if w == 0 or h == 0:
            return None
        return ScreenRegion(label, int(x), int(y), int(w), int(h))

    print("1) Select the PRICE region (the last-traded price label).")
    cfg.price_region = select("price")

    print("2) Select the CHART region (the candlestick/price-action area).")
    cfg.chart_region = select("chart")

    cfg.indicator_regions = {}
    print("3) Optionally select additional indicator regions (e.g. RSI readout). Leave name blank to finish.")
    while True:
        name = input("   Indicator name (blank to finish): ").strip()
        if not name:
            break
        region = select(name)
        if region is not None:
            cfg.indicator_regions[name] = region

    cfg.save()
    print(f"Saved calibration to {ScreenConfig.calibration_path()}")


def test_extraction():
    """Quick sanity check after calibration: prints what extract_price /
    extract_indicators currently read off your screen."""
    cfg = ScreenConfig.load()
    try:
        price = extract_price(cfg)
        print(f"price_region OCR -> {price}")
    except Exception as e:
        print(f"price_region OCR failed: {e}")
    if cfg.indicator_regions:
        print(f"indicator_regions OCR -> {extract_indicators(cfg)}")


if __name__ == "__main__":
    import sys

    if "--calibrate" in sys.argv:
        calibrate()
    elif "--test" in sys.argv:
        test_extraction()
    else:
        print("Usage: python screen_vision.py --calibrate | --test")
