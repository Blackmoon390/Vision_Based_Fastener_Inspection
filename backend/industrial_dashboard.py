"""
industrial_dashboard.py  —  v8  (overlay mode)
===============================================
The live camera frame fills the ENTIRE 1920×1080 canvas (letterboxed to fit).
All UI panels are drawn ON TOP with semi-transparent backgrounds — nothing
is outside the frame, no black side-columns.

Layout (overlaid on live feed):
  TOP-LEFT      : ORIENTATION MASK
  LEFT          : MEASUREMENT DATA + TOLERANCE STATUS
  TOP-RIGHT     : BOLT MASK
  RIGHT         : SYSTEM INFORMATION + FPS METER
  BOTTOM-RIGHT  : INSPECTION STATUS (PASS/FAIL)

render_dashboard(measurement_data, live_frame=None) -> np.ndarray (1080,1920,3)
init_window(name)
show_frame(img, name) -> int   — ONLY ONE imshow ever called
"""

import cv2
import numpy as np
import datetime
from typing import Optional

# ─────────────────────────────────────────────────────────────
#  COLOUR PALETTE  (BGR)
# ─────────────────────────────────────────────────────────────
BG        = ( 20,  20,  20)
PANEL_BG  = ( 30,  30,  30)      # semi-transparent overlay base
CYAN      = (255, 220,   0)
TEXT      = (255, 255, 255)
GREEN     = (  0, 220,   0)
RED       = (  0,   0, 220)
ORANGE    = (  0, 155, 255)
DIM       = (160, 160, 160)
SEP       = ( 70,  70,  70)
GREY_BDR  = (180, 180, 180)
ALPHA     = 0.78                 # panel opacity over live feed

DASH_W, DASH_H = 1920, 1080

# ─────────────────────────────────────────────────────────────
#  ALERT BUTTON  — region stored for mouse-hit detection
# ─────────────────────────────────────────────────────────────
import threading

_ALERT_BTN_RECT: tuple = (0, 0, 0, 0)   # (x1, y1, x2, y2) filled at render time

# Event that VBAFI (or any caller) can wait on.
# Set each time the button is clicked; caller clears it after handling.
alert_event: threading.Event = threading.Event()

# Optional extra callback registered by the caller (e.g. VBAFI).
_alert_callback = None

def register_alert_callback(fn) -> None:
    """Call this from optimized_VBAFI.py to hook the button click.

    Example::
        import industrial_dashboard as dash
        dash.register_alert_callback(my_trigger_fn)
    """
    global _alert_callback
    _alert_callback = fn

def on_alert_click() -> None:
    """Called when the ALERT button is clicked."""
    print("click generate report")
    alert_event.set()                   # signal any thread waiting on the event
    if _alert_callback is not None:
        _alert_callback()               # call the registered VBAFI handler

BASE_W, BASE_H = 1920, 1080

WINDOW_SCALE = 1.0   # default

DASH_W = int(BASE_W * WINDOW_SCALE)
DASH_H = int(BASE_H * WINDOW_SCALE)


PANEL_W   = 300
PANEL_PAD = 10
TITLE_H   = 28
BORDER    = 1

LEFT_X1  = 8
LEFT_X2  = LEFT_X1 + PANEL_W
RIGHT_X2 = DASH_W - 8
RIGHT_X1 = RIGHT_X2 - PANEL_W

_F = cv2.FONT_HERSHEY_SIMPLEX

def set_window_scale(scale: float):
    global WINDOW_SCALE, DASH_W, DASH_H
    global LEFT_X1, LEFT_X2, RIGHT_X1, RIGHT_X2

    WINDOW_SCALE = scale
    DASH_W = int(BASE_W * scale)
    DASH_H = int(BASE_H * scale)

    # Recompute layout constants that depend on DASH_W
    LEFT_X1  = 8
    LEFT_X2  = LEFT_X1 + PANEL_W
    RIGHT_X2 = DASH_W - 8
    RIGHT_X1 = RIGHT_X2 - PANEL_W





# ═════════════════════════════════════════════════════════════
#  WINDOW HELPERS  — single window, fullscreen
# ═════════════════════════════════════════════════════════════

def _mouse_callback(event, x, y, flags, param) -> None:
    """OpenCV mouse callback — fires on_alert_click when button region is hit."""
    if event == cv2.EVENT_LBUTTONDOWN:
        bx1, by1, bx2, by2 = _ALERT_BTN_RECT
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            on_alert_click()


def init_window(name: str = "BOLT INSPECTION SYSTEM") -> None:
    import sys, ctypes
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, DASH_W, DASH_H)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(name, _mouse_callback)


def show_frame(img: np.ndarray,
               name: str = "BOLT INSPECTION SYSTEM") -> int:
    """The ONE AND ONLY imshow call in the whole system."""
    cv2.imshow(name, img)
    return cv2.waitKey(1) & 0xFF


# ═════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════

def _txt(canvas, text, x, y, color=TEXT, scale=0.44, thick=1):
    cv2.putText(canvas, text, (x, y), _F, scale, color, thick, cv2.LINE_AA)


def _blend_rect(canvas: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                color: tuple, alpha: float) -> None:
    """Draw a semi-transparent filled rectangle onto canvas."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(canvas.shape[1], x2), min(canvas.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = canvas[y1:y2, x1:x2].astype(np.float32)
    overlay = np.full_like(roi, color, dtype=np.float32)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)
    canvas[y1:y2, x1:x2] = roi.astype(np.uint8)


def draw_panel(canvas: np.ndarray,
               x1: int, y1: int, x2: int, y2: int,
               title: str,
               border_color: tuple = CYAN,
               alpha: float = ALPHA) -> int:
    """Blended panel. Returns Y of content start."""
    _blend_rect(canvas, x1, y1, x2, y2, PANEL_BG, alpha)
    # Title bar — fully opaque
    cv2.rectangle(canvas, (x1, y1), (x2, y1 + TITLE_H), border_color, -1)
    _txt(canvas, title, x1 + PANEL_PAD, y1 + TITLE_H - 7,
         color=(10, 10, 10), scale=0.48, thick=1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, BORDER)
    return y1 + TITLE_H + PANEL_PAD


# ═════════════════════════════════════════════════════════════
#  LIVE FEED — fills entire 1920×1080, letterboxed
# ═════════════════════════════════════════════════════════════

def draw_live_feed(canvas: np.ndarray,
                   live_frame: Optional[np.ndarray]) -> None:
    """
    Scale live_frame to fit inside DASH_W × DASH_H keeping aspect ratio.
    Black bars fill any unused edges. Works for ANY source resolution.
    """
    if live_frame is None or live_frame.size == 0:
        canvas[:] = BG
        _txt(canvas, "NO LIVE FEED",
             DASH_W // 2 - 100, DASH_H // 2,
             DIM, scale=1.2, thick=2)
        return

    canvas[:] = (0, 0, 0)          # black letterbox background

    src_h, src_w = live_frame.shape[:2]
    scale = min(DASH_W / src_w, DASH_H / src_h)
    dst_w = int(src_w * scale)
    dst_h = int(src_h * scale)
    off_x = (DASH_W - dst_w) // 2
    off_y = (DASH_H - dst_h) // 2

    resized = cv2.resize(live_frame, (dst_w, dst_h),
                         interpolation=cv2.INTER_LINEAR)
    canvas[off_y:off_y + dst_h, off_x:off_x + dst_w] = resized


# ═════════════════════════════════════════════════════════════
#  TOP-LEFT: ORIENTATION MASK
# ═════════════════════════════════════════════════════════════

def draw_orientation_mask_panel(canvas: np.ndarray,
                                orientation_mask: Optional[np.ndarray]) -> None:
    PX1, PY1 = LEFT_X1, 8
    PX2, PY2 = LEFT_X2, 248
    content_y = draw_panel(canvas, PX1, PY1, PX2, PY2,
                            "ORIENTATION MASK", border_color=CYAN)
    ix1 = PX1 + PANEL_PAD;  iy1 = content_y
    ix2 = PX2 - PANEL_PAD;  iy2 = PY2 - PANEL_PAD
    iw, ih = ix2 - ix1, iy2 - iy1

    if iw <= 0 or ih <= 0:
        return  # panel too small to render

    if orientation_mask is not None and orientation_mask.size > 0:
        disp = cv2.cvtColor(orientation_mask, cv2.COLOR_GRAY2BGR) \
               if len(orientation_mask.shape) == 2 else orientation_mask.copy()
        canvas[iy1:iy2, ix1:ix2] = cv2.resize(disp, (iw, ih),
                                               interpolation=cv2.INTER_LINEAR)
    else:
        cv2.rectangle(canvas, (ix1, iy1), (ix2, iy2), (50, 50, 50), -1)
        _txt(canvas, "NO DATA", ix1 + iw // 2 - 34, iy1 + ih // 2 + 6, DIM, 0.44)


# ═════════════════════════════════════════════════════════════
#  LEFT-CENTER: MEASUREMENT DATA
# ═════════════════════════════════════════════════════════════

def draw_measurement_panel(canvas: np.ndarray,
                           bolt_height: float, thread_height: float,
                           bolt_total_width: float, bolt_center_width: float,
                           bolt_bottom_width: float, head_diameter: float,
                           thread_diameter: float) -> int:
    """Returns bottom Y of this panel."""
    PX1, PY1 = LEFT_X1, 256
    PX2       = LEFT_X2
    ROW_H     = 34
    PY2       = PY1 + TITLE_H + PANEL_PAD + 7 * ROW_H + PANEL_PAD

    content_y = draw_panel(canvas, PX1, PY1, PX2, PY2,
                            "MEASUREMENT DATA", border_color=CYAN)

    rows = [
        ("BOLT HEIGHT",     f"{bolt_height:.2f} mm"),
        ("THREAD HEIGHT",   f"{thread_height:.2f} mm"),
        ("TOTAL WIDTH",     f"{bolt_total_width:.2f} mm"),
        ("CENTER WIDTH",    f"{bolt_center_width:.2f} mm"),
        ("BOTTOM WIDTH",    f"{bolt_bottom_width:.2f} mm"),
        ("HEAD DIAMETER",   f"{head_diameter:.2f} mm"),
        ("THREAD DIAMETER", f"{thread_diameter:.2f} mm"),
    ]
    for i, (label, value) in enumerate(rows):
        ry = content_y + i * ROW_H
        if i % 2 == 0:
            _blend_rect(canvas, PX1 + 1, ry - 4, PX2 - 1, ry + ROW_H - 8,
                        (50, 50, 50), 0.5)
        _txt(canvas, label, PX1 + PANEL_PAD, ry + 7,  DIM,  0.35)
        _txt(canvas, value, PX1 + PANEL_PAD, ry + 24, TEXT, 0.50)
        cv2.line(canvas, (PX1 + 4, ry + ROW_H - 8),
                 (PX2 - 4, ry + ROW_H - 8), SEP, 1)
    return PY2


# ═════════════════════════════════════════════════════════════
#  LEFT-BOTTOM: TOLERANCE STATUS
# ═════════════════════════════════════════════════════════════

def draw_tolerance_panel(canvas: np.ndarray,
                         bolt_height: float, thread_height: float,
                         bolt_total_width: float, bolt_center_width: float,
                         bolt_bottom_width: float, head_diameter: float,
                         thread_diameter: float,
                         top_y: int) -> None:
    PX1, PX2 = LEFT_X1, LEFT_X2
    PY1 = top_y + 8
    PY2 = DASH_H - 8

    content_y = draw_panel(canvas, PX1, PY1, PX2, PY2,
                            "TOLERANCE STATUS", border_color=CYAN)

    TOLERANCES = [
        ("BOLT HEIGHT",   bolt_height,       25.0, 35.0),
        ("THREAD HEIGHT", thread_height,     20.0, 30.0),
        ("TOTAL WIDTH",   bolt_total_width,  10.0, 14.0),
        ("CENTER WIDTH",  bolt_center_width,  5.0,  8.0),
        ("BOTTOM WIDTH",  bolt_bottom_width,  5.0,  8.0),
        ("HEAD DIA",      head_diameter,     10.0, 14.0),
        ("THREAD DIA",    thread_diameter,    5.0,  8.0),
    ]

    avail_h = PY2 - content_y - 8
    ROW_H   = max(26, min(48, avail_h // len(TOLERANCES)))
    BAR_W   = PX2 - PX1 - PANEL_PAD * 2
    BAR_H   = 7

    for i, (label, value, lo, hi) in enumerate(TOLERANCES):
        ry = content_y + i * ROW_H
        if ry + ROW_H > PY2 - 4:
            break
        if i % 2 == 0:
            _blend_rect(canvas, PX1 + 1, ry - 2, PX2 - 1, ry + ROW_H - 4,
                        (50, 50, 50), 0.5)
        in_tol    = lo <= value <= hi
        dot_color = GREEN if in_tol else RED
        val_col   = GREEN if in_tol else RED
        cv2.circle(canvas, (PX1 + PANEL_PAD + 5, ry + 8), 5, dot_color, -1)
        _txt(canvas, label,          PX1 + PANEL_PAD + 16, ry + 10, DIM,     0.34)
        _txt(canvas, f"{value:.2f}", PX2 - 60,              ry + 10, val_col, 0.40)

        bx1 = PX1 + PANEL_PAD; bx2 = bx1 + BAR_W
        by  = ry + ROW_H - 12
        cv2.rectangle(canvas, (bx1, by), (bx2, by + BAR_H), (55, 55, 55), -1)
        span   = hi - lo if hi != lo else 1
        t      = max(0.0, min(1.0, (value - lo) / span))
        mark_x = bx1 + int(t * BAR_W)
        cv2.rectangle(canvas, (bx1, by), (mark_x, by + BAR_H), dot_color, -1)
        cv2.rectangle(canvas, (bx1, by), (bx2, by + BAR_H), (80, 80, 80), 1)
        _txt(canvas, f"{lo:.0f}", bx1,      by + BAR_H + 9, DIM, 0.27)
        _txt(canvas, f"{hi:.0f}", bx2 - 16, by + BAR_H + 9, DIM, 0.27)
        cv2.line(canvas, (PX1 + 4, ry + ROW_H - 2),
                 (PX2 - 4, ry + ROW_H - 2), SEP, 1)


# ═════════════════════════════════════════════════════════════
#  TOP-RIGHT: BOLT MASK
# ═════════════════════════════════════════════════════════════

def draw_bolt_mask_panel(canvas: np.ndarray,
                         bolt_mask: Optional[np.ndarray]) -> None:
    PX1, PY1 = RIGHT_X1, 8
    PX2, PY2 = RIGHT_X2, 248
    content_y = draw_panel(canvas, PX1, PY1, PX2, PY2,
                            "BOLT MASK", border_color=GREY_BDR)
    ix1 = PX1 + PANEL_PAD;  iy1 = content_y
    ix2 = PX2 - PANEL_PAD;  iy2 = PY2 - PANEL_PAD
    iw, ih = ix2 - ix1, iy2 - iy1

    if iw <= 0 or ih <= 0:
        return  # panel too small to render (e.g. extreme scale values)

    if bolt_mask is not None and bolt_mask.size > 0:
        disp = cv2.cvtColor(bolt_mask, cv2.COLOR_GRAY2BGR) \
               if len(bolt_mask.shape) == 2 else bolt_mask.copy()
        canvas[iy1:iy2, ix1:ix2] = cv2.resize(disp, (iw, ih),
                                               interpolation=cv2.INTER_LINEAR)
    else:
        cv2.rectangle(canvas, (ix1, iy1), (ix2, iy2), (50, 50, 50), -1)
        _txt(canvas, "NO DATA", ix1 + iw // 2 - 34, iy1 + ih // 2 + 6, DIM, 0.44)


# ═════════════════════════════════════════════════════════════
#  RIGHT-CENTER: SYSTEM INFORMATION
# ═════════════════════════════════════════════════════════════

def draw_system_info_panel(canvas: np.ndarray,
                           fps: float, gpu_status: str,
                           camera_status: str, frame_id: int,
                           inference_time_ms: float) -> int:
    PX1, PY1 = RIGHT_X1, 256
    PX2       = RIGHT_X2
    ROW_H     = 34
    PY2       = PY1 + TITLE_H + PANEL_PAD + 7 * ROW_H + PANEL_PAD

    content_y = draw_panel(canvas, PX1, PY1, PX2, PY2,
                            "SYSTEM INFORMATION", border_color=GREY_BDR)

    now  = datetime.datetime.now()
    rows = [
        ("FPS",       f"{fps:.0f}",
         GREEN if fps > 25 else ORANGE if fps >= 15 else RED),
        ("GPU",       gpu_status,
         GREEN if gpu_status == "ACTIVE" else ORANGE),
        ("CAMERA",    camera_status,
         GREEN if camera_status == "ONLINE" else RED),
        ("FRAME ID",  f"{frame_id}",                TEXT),
        ("TIME",      now.strftime("%H:%M:%S"),      TEXT),
        ("DATE",      now.strftime("%d-%m-%Y"),      TEXT),
        ("INFERENCE", f"{inference_time_ms:.0f} ms",
         GREEN if inference_time_ms < 30 else ORANGE),
    ]
    for i, (label, value, vcol) in enumerate(rows):
        ry = content_y + i * ROW_H
        if i % 2 == 0:
            _blend_rect(canvas, PX1 + 1, ry - 4, PX2 - 1, ry + ROW_H - 8,
                        (50, 50, 50), 0.5)
        _txt(canvas, label, PX1 + PANEL_PAD, ry + 7,  DIM,  0.35)
        _txt(canvas, value, PX1 + PANEL_PAD, ry + 24, vcol, 0.50)
        cv2.line(canvas, (PX1 + 4, ry + ROW_H - 8),
                 (PX2 - 4, ry + ROW_H - 8), SEP, 1)
    return PY2


# ═════════════════════════════════════════════════════════════
#  RIGHT: VERTICAL FPS METER  (between sys-info and status)
# ═════════════════════════════════════════════════════════════

def draw_fps_meter(canvas: np.ndarray, fps: float,
                   top_y: int, bottom_y: int) -> None:
    PX1, PX2 = RIGHT_X1, RIGHT_X2
    PY1 = top_y + 8
    PY2 = bottom_y - 8
    if PY2 - PY1 < 40:
        return

    _blend_rect(canvas, PX1, PY1, PX2, PY2, PANEL_BG, ALPHA)
    cv2.rectangle(canvas, (PX1, PY1), (PX2, PY2), GREY_BDR, BORDER)
    _txt(canvas, "FPS MONITOR", PX1 + PANEL_PAD, PY1 + 18, DIM, 0.42)
    cv2.line(canvas, (PX1 + 4, PY1 + 24), (PX2 - 4, PY1 + 24), SEP, 1)

    BAR_W = 34
    cx    = PX1 + (PX2 - PX1) // 2
    bx1, bx2 = cx - BAR_W // 2, cx + BAR_W // 2
    by1 = PY1 + 32
    by2 = PY2 - 28
    if by2 - by1 < 20:
        return
    BAR_H = by2 - by1

    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (50, 50, 50), -1)
    fps_c  = max(0.0, min(100.0, fps))
    fill_h = int(BAR_H * fps_c / 100.0)
    bar_col = GREEN if fps_c > 25 else (ORANGE if fps_c >= 15 else RED)
    if fill_h > 0:
        cv2.rectangle(canvas, (bx1, by2 - fill_h), (bx2, by2), bar_col, -1)
    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (100, 100, 100), 1)

    for tick in [0, 25, 50, 75, 100]:
        ty = by2 - int(BAR_H * tick / 100.0)
        cv2.line(canvas, (bx1 - 6, ty), (bx1, ty), (150, 150, 150), 1)
        _txt(canvas, str(tick), bx2 + 4, ty + 4, DIM, 0.30)

    fps_str = f"{fps:.0f}"
    (tw, _), _ = cv2.getTextSize(fps_str, _F, 0.85, 2)
    ty = by2 + 16
    if ty < PY2 - 4:
        cv2.putText(canvas, fps_str, (cx - tw // 2, ty),
                    _F, 0.85, bar_col, 2, cv2.LINE_AA)
        _txt(canvas, "FPS", cx - 12, ty + 16, DIM, 0.34)


# ═════════════════════════════════════════════════════════════
#  BOTTOM-RIGHT: PASS / FAIL STATUS
# ═════════════════════════════════════════════════════════════

def draw_status_panel(canvas: np.ndarray, status: str) -> int:
    BOX_H = 90
    BX1, BX2 = RIGHT_X1, RIGHT_X2
    BY2 = DASH_H - 8
    BY1 = BY2 - BOX_H

    status_up = (status or "").upper()
    color = GREEN if status_up == "PASS" else (RED if status_up == "FAIL" else DIM)

    _blend_rect(canvas, BX1, BY1, BX2, BY2, PANEL_BG, ALPHA)
    cv2.rectangle(canvas, (BX1, BY1), (BX2, BY2), color, 3)
    _txt(canvas, "INSPECTION STATUS", BX1 + PANEL_PAD, BY1 + 15, DIM, 0.34)

    (tw, th), _ = cv2.getTextSize(status_up, _F, 1.3, 3)
    tx = BX1 + (BX2 - BX1 - tw) // 2
    ty = BY1 + 20 + (BOX_H - 20 + th) // 2
    cv2.putText(canvas, status_up, (tx, ty), _F, 1.3, color, 3, cv2.LINE_AA)
    return BY1


# ═════════════════════════════════════════════════════════════
#  LEFT-BOTTOM-RIGHT: GENERATE REPORT BUTTON  (below-right of tolerance box)
# ═════════════════════════════════════════════════════════════

def draw_alert_button(canvas: np.ndarray) -> None:
    """Draw a clickable GENERATE REPORT button just to the bottom-right of the
    tolerance panel.  Updates the global _ALERT_BTN_RECT so the mouse
    callback knows where to hit-test."""
    global _ALERT_BTN_RECT

    BTN_W, BTN_H = 180, 50
    PAD           = 6

    # Anchor: right edge of the left panel column, bottom of screen
    bx2 = LEFT_X2 + BTN_W + PAD          # right edge of button
    bx1 = bx2 - BTN_W                    # left  edge
    by2 = DASH_H - 8                     # bottom (same as tolerance panel)
    by1 = by2 - BTN_H                    # top

    _ALERT_BTN_RECT = (bx1, by1, bx2, by2)

    # Button body — green fill with green border
    _blend_rect(canvas, bx1, by1, bx2, by2, (0, 120, 0), 0.90)
    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), GREEN, 2)

    # Centred label
    label = "GENERATE REPORT"
    (tw, th), _ = cv2.getTextSize(label, _F, 0.52, 1)
    tx = bx1 + (BTN_W - tw) // 2
    ty = by1 + (BTN_H + th) // 2
    cv2.putText(canvas, label, (tx, ty), _F, 0.52, TEXT, 1, cv2.LINE_AA)


# ═════════════════════════════════════════════════════════════
#  MAIN RENDER FUNCTION
# ═════════════════════════════════════════════════════════════

def render_dashboard(measurement_data: dict,
                     live_frame: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Returns a 1920×1080 BGR frame.

    The live_frame is scaled to fill the full 1920×1080 (letterboxed).
    All UI panels are blended ON TOP — nothing is outside the frame.

    measurement_data keys:
        bolt_height, thread_height, bolt_total_width,
        bolt_center_width, bolt_bottom_width,
        head_diameter, thread_diameter,
        bolt_mask, orientation_mask,
        fps, gpu_status, camera_status,
        frame_id, inference_time_ms,
        status  ("PASS" | "FAIL")
    """
    canvas = np.zeros((DASH_H, DASH_W, 3), dtype=np.uint8)

    # ── 1. Live feed fills full canvas ───────────────────────
    draw_live_feed(canvas, live_frame)

    # ── 2. Extract values ────────────────────────────────────
    bh  = float(measurement_data.get("bolt_height",       0.0))
    th  = float(measurement_data.get("thread_height",     0.0))
    btw = float(measurement_data.get("bolt_total_width",  0.0))
    bcw = float(measurement_data.get("bolt_center_width", 0.0))
    bbw = float(measurement_data.get("bolt_bottom_width", 0.0))
    hd  = float(measurement_data.get("head_diameter",     0.0))
    td  = float(measurement_data.get("thread_diameter",   0.0))
    fps = float(measurement_data.get("fps",               0.0))

    # ── 3. Draw panels ON TOP ────────────────────────────────
    draw_orientation_mask_panel(canvas, measurement_data.get("orientation_mask"))
    meas_bottom = draw_measurement_panel(canvas, bh, th, btw, bcw, bbw, hd, td)
    draw_tolerance_panel(canvas, bh, th, btw, bcw, bbw, hd, td, meas_bottom)
    draw_alert_button(canvas)

    draw_bolt_mask_panel(canvas, measurement_data.get("bolt_mask"))
    sys_bottom = draw_system_info_panel(
        canvas,
        fps               = fps,
        gpu_status        = measurement_data.get("gpu_status",        "UNKNOWN"),
        camera_status     = measurement_data.get("camera_status",     "OFFLINE"),
        frame_id          = measurement_data.get("frame_id",          0),
        inference_time_ms = measurement_data.get("inference_time_ms", 0.0),
    )
    status_top = draw_status_panel(canvas, measurement_data.get("status", "--"))
    draw_fps_meter(canvas, fps, top_y=sys_bottom, bottom_y=status_top)

    return canvas


# ═════════════════════════════════════════════════════════════
#  DEMO / SELF-TEST
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    bolt_mask = np.zeros((200, 100), dtype=np.uint8)
    cv2.ellipse(bolt_mask, (50, 100), (35, 90), 0, 0, 360, 255, -1)

    orient_mask = np.zeros((200, 100), dtype=np.uint8)
    cv2.ellipse(orient_mask, (50, 100), (35, 90), 0, 0, 360, 255, -1)
    cv2.line(orient_mask, (50, 10), (50, 190), 160, 2)

    # Simulate a 4K camera frame
    fake_4k = np.zeros((2160, 3840, 3), dtype=np.uint8)
    for r in range(2160):
        fake_4k[r, :, 0] = int(r / 2160 * 60)
        fake_4k[r, :, 1] = int(r / 2160 * 80)
        fake_4k[r, :, 2] = int(r / 2160 * 40)
    cv2.putText(fake_4k, "4K LIVE FEED", (300, 1080),
                cv2.FONT_HERSHEY_SIMPLEX, 8, (200, 200, 200), 12)

    scenarios = [(30.0, "PASS"), (22.0, "PASS"), (10.0, "FAIL"), (30.0, "PASS")]
    init_window()

    for cycle in range(400):
        fps_val, status = scenarios[(cycle // 100) % len(scenarios)]
        data = {
            "bolt_height": 30.00, "thread_height": 25.00,
            "bolt_total_width": 12.00, "bolt_center_width": 6.20,
            "bolt_bottom_width": 6.00, "head_diameter": 12.10,
            "thread_diameter": 6.05,
            "bolt_mask": bolt_mask, "orientation_mask": orient_mask,
            "fps": fps_val, "gpu_status": "ACTIVE",
            "camera_status": "ONLINE", "frame_id": 1452 + cycle,
            "inference_time_ms": 22.0, "status": status,
        }
        frame = render_dashboard(data, live_frame=fake_4k)
        key   = show_frame(frame)
        if key == ord("q"):
            break
        time.sleep(1 / 30)

    cv2.destroyAllWindows()
