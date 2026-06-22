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
import pytesseract


logger = logging.getLogger(__name__)

_NUMERIC_OCR_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789.,-"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def capture_region(region: ScreenRegion) -> np.ndarray:
    """Grab one region of the screen as a BGR numpy array."""
    import mss
    import cv2

    with mss.MSS() as sct:          # mss.mss() is deprecated; MSS is the current API
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


# ---------------------------------------------------------------------------
# Calibration -- tkinter-based region selector (no OpenCV GUI required)
# ---------------------------------------------------------------------------

def _take_full_screenshot():
    """Capture the primary monitor and return it as an RGB PIL Image.

    Uses mss.MSS (non-deprecated API). The raw BGRA bytes from mss are
    converted to RGB via numpy before PIL receives them so no OpenCV call
    is needed here either.
    """
    import mss
    from PIL import Image

    with mss.MSS() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        # np.array(shot) → shape (height, width, 4), channel order BGRA.
        # Drop alpha ([:, :, :3]) then reverse channels ([:, :, ::-1]) → RGB.
        img_np = np.array(shot)[:, :, :3][:, :, ::-1]
    return Image.fromarray(img_np, "RGB")


def _select_region_tk(label: str, screenshot_pil) -> "ScreenRegion | None":
    """Show *screenshot_pil* in a tkinter window and let the user drag a
    selection rectangle over the region they want to calibrate.

    This replaces cv2.selectROI so --calibrate works with every OpenCV
    build, including opencv-python-headless (which has no GUI functions).
    tkinter ships with every standard CPython / Anaconda installation, so
    no extra package is required.

    Controls
    --------
    Left-click + drag   draw / redraw the selection box
    Enter  or  Space    confirm the box and close
    Esc    or  c        skip this region (returns None)
    """
    import tkinter as tk
    from PIL import Image as _PILImage, ImageTk

    _result = [None]   # list so the nested closures can write back

    root = tk.Tk()
    root.title(
        f"Select '{label}'  |  drag to draw  |"
        "  Enter/Space = confirm  |  Esc/c = skip"
    )
    root.resizable(False, False)

    # ---- scale the screenshot to fit 90 % of the display ----------------
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    orig_w, orig_h = screenshot_pil.size
    scale = min(0.90 * screen_w / orig_w, 0.90 * screen_h / orig_h, 1.0)
    disp_w = int(orig_w * scale)
    disp_h = int(orig_h * scale)

    # Pillow ≥10 moved resampling filters under Image.Resampling; stay
    # compatible with the older versions still common in Anaconda.
    try:
        _LANCZOS = _PILImage.Resampling.LANCZOS
    except AttributeError:
        _LANCZOS = _PILImage.LANCZOS  # type: ignore[attr-defined]

    img_disp = screenshot_pil.resize((disp_w, disp_h), _LANCZOS)
    tk_img = ImageTk.PhotoImage(img_disp)

    # ---- canvas ----------------------------------------------------------
    canvas = tk.Canvas(
        root, width=disp_w, height=disp_h,
        cursor="crosshair", highlightthickness=0,
    )
    canvas.pack()
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

    # ---- status bar ------------------------------------------------------
    status_var = tk.StringVar(value=f"Drag to select '{label}'")
    tk.Label(
        root, textvariable=status_var,
        bg="#1e1e1e", fg="#d4d4d4", anchor=tk.W, padx=6, pady=3,
    ).pack(fill=tk.X)

    # ---- interaction state ----------------------------------------------
    _st: dict = {"start": None, "rect_id": None, "coords": None}

    def on_press(e):
        _st["start"] = (e.x, e.y)
        if _st["rect_id"]:
            canvas.delete(_st["rect_id"])
            _st["rect_id"] = None

    def on_drag(e):
        if _st["start"] is None:
            return
        if _st["rect_id"]:
            canvas.delete(_st["rect_id"])
        x0, y0 = _st["start"]
        _st["rect_id"] = canvas.create_rectangle(
            x0, y0, e.x, e.y,
            outline="#ff4444", width=2, dash=(6, 3),
        )
        w_disp = abs(e.x - x0)
        h_disp = abs(e.y - y0)
        status_var.set(
            f"Drawing '{label}': {int(w_disp / scale)} × {int(h_disp / scale)}"
            " screen px  |  release to finalize"
        )

    def on_release(e):
        if _st["start"] is None:
            return
        x0, y0 = _st["start"]
        x1, y1 = e.x, e.y
        x = min(x0, x1)
        y = min(y0, y1)
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        if w > 2 and h > 2:
            # Map display-pixel coords back to real screen coordinates
            _st["coords"] = (
                int(x / scale), int(y / scale),
                int(w / scale), int(h / scale),
            )
            status_var.set(
                f"Selected: {int(w / scale)} × {int(h / scale)} px  |  "
                "Enter/Space = confirm  |  Esc/c = skip"
            )
        else:
            status_var.set("Box too small — drag a larger area and release.")

    def confirm(e=None):
        _result[0] = _st["coords"]
        root.destroy()

    def cancel(e=None):
        _result[0] = None
        root.destroy()

    canvas.bind("<ButtonPress-1>",   on_press)
    canvas.bind("<B1-Motion>",       on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Return>", confirm)
    root.bind("<space>",  confirm)
    root.bind("<Escape>", cancel)
    root.bind("c",        cancel)

    root.mainloop()

    if _result[0] is None:
        return None
    x, y, w, h = _result[0]
    return ScreenRegion(label, x, y, w, h)


def calibrate():
    """Interactive setup: screenshot the primary monitor, then drag boxes
    over the price label, the chart area, and any indicator readouts.

    Uses a tkinter window for region selection -- works with every OpenCV
    build including opencv-python-headless, because no cv2 GUI call is
    made here.  tkinter is part of the Python standard library and is
    included in every Anaconda distribution, so no extra install is needed.
    """
    print(
        "Make sure your trading platform is visible in its normal position.\n"
        "A screenshot will be taken when you press Enter."
    )
    input("Press Enter to capture... ")

    screenshot = _take_full_screenshot()

    cfg = ScreenConfig()
    cfg.monitor_index = 1

    print("1) Select the PRICE region (the last-traded price label).")
    cfg.price_region = _select_region_tk("price", screenshot)

    print("2) Select the CHART region (the candlestick/price-action area).")
    cfg.chart_region = _select_region_tk("chart", screenshot)

    cfg.indicator_regions = {}
    print(
        "3) Optionally select additional indicator regions (e.g. RSI readout).\n"
        "   Leave the name blank to finish."
    )
    while True:
        name = input("   Indicator name (blank to finish): ").strip()
        if not name:
            break
        region = _select_region_tk(name, screenshot)
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