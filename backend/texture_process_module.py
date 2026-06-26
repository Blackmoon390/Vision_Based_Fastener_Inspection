import cv2
import numpy as np
import math
from collections import deque

# =========================================================
# COLOUR PROCESS
# =========================================================

def best_edge_fix(img_path):

    img = img_path

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(gray)

    # SPEED: reduced d from 9 -> 5 (bilateralFilter cost is O(d^2))
    gray = cv2.bilateralFilter(
        gray,
        5,
        75,
        75
    )

    kernel_grad = np.ones((5, 5), np.uint8)

    grad = cv2.morphologyEx(
        gray,
        cv2.MORPH_GRADIENT,
        kernel_grad
    )

    return gray, grad


# =========================================================
# PCA / ROTATION
# =========================================================

def keep_largest_component(mask):
    """
    Keep only largest connected object
    """

    mask = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = \
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )

    if num_labels <= 1:
        return mask

    largest_label = 1 + np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    )

    clean = np.zeros_like(mask)

    clean[labels == largest_label] = 1

    return clean


def get_largest_contour(mask):
    """
    Get largest contour
    SPEED: CHAIN_APPROX_SIMPLE instead of CHAIN_APPROX_NONE
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE   # was CHAIN_APPROX_NONE — stores fewer points
    )

    if len(contours) == 0:
        return None

    return max(contours, key=cv2.contourArea)


def get_mask_center(mask):
    """
    Get center of object
    """

    contour = get_largest_contour(mask)

    if contour is None:
        return None

    M = cv2.moments(contour)

    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return (cx, cy)


def rotate_mask(mask, angle, center=None):
    """
    Rotate mask without clipping.
    SPEED: tight diagonal pad instead of max(h,w) on all 4 sides.
    """

    mask = (mask > 0).astype(np.uint8)

    h, w = mask.shape[:2]

    # Tight pad: only need half the diagonal to avoid clipping
    pad = int(math.hypot(h, w) / 2) + 2

    padded = cv2.copyMakeBorder(
        mask,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=0
    )

    ph, pw = padded.shape[:2]

    if center is None:
        center = (pw // 2, ph // 2)

    else:
        center = (
            center[0] + pad,
            center[1] + pad
        )

    M = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        padded,
        M,
        (pw, ph),
        flags=cv2.INTER_NEAREST,
        borderValue=0
    )

    return rotated


def get_mask_height(mask):
    """
    Get vertical height
    """

    contour = get_largest_contour(mask)

    if contour is None:
        return 0

    x, y, w, h = cv2.boundingRect(contour)

    return h


def get_orientation(mask):
    """
    PCA orientation
    """

    contour = get_largest_contour(mask)

    if contour is None:
        return 0

    coords = contour.reshape(-1, 2).astype(np.float32)

    mean, eigenvectors = cv2.PCACompute(
        coords,
        mean=None
    )

    vx, vy = eigenvectors[0]

    angle = np.degrees(
        np.arctan2(vy, vx)
    )

    return angle


def get_top_bottom_points(mask):
    """
    Returns:
        top_point
        bottom_point
    """

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None, None

    min_y = ys.min()
    max_y = ys.max()

    top_xs = xs[ys == min_y]
    bottom_xs = xs[ys == max_y]

    top_point = (
        int(np.mean(top_xs)),
        int(min_y)
    )

    bottom_point = (
        int(np.mean(bottom_xs)),
        int(max_y)
    )

    return top_point, bottom_point


def _is_upside_down(mask, band_percent=0.25):
    """
    Determine whether the wider/heavier section of the object sits
    at the bottom rather than the top.

    The mask coming in may have large amounts of black padding around
    the object (added by rotate_mask). Slicing raw rows therefore hits
    empty space instead of the object ends, so we MUST crop tightly to
    the object bounding box before sampling the bands.

    Strategy
    --------
    1. Crop to the object bounding box (eliminates padding rows).
    2. Sample the top `band_percent` of the cropped height → max width.
    3. Sample the bottom `band_percent` of the cropped height → max width.
    4. If bottom max-width > top max-width the object is upside-down.

    Parameters
    ----------
    mask         : uint8 ndarray  — binary mask (any positive value = object)
    band_percent : float          — fraction of OBJECT height to sample
                                    at each end (default 25 %)

    Returns
    -------
    bool  — True if a 180-degree flip is needed to put the wider
            section at the top.
    """

    mask_bin = (mask > 0).astype(np.uint8)

    # ── tight crop to object bounding box ────────────────────────────
    ys, xs = np.where(mask_bin > 0)

    if len(ys) == 0:
        return False                        # empty mask — nothing to flip

    y1_obj, y2_obj = int(ys.min()), int(ys.max())
    x1_obj, x2_obj = int(xs.min()), int(xs.max())

    cropped = mask_bin[y1_obj : y2_obj + 1,
                       x1_obj : x2_obj + 1]   # object fills this canvas

    h, w   = cropped.shape
    band_h = max(1, int(h * band_percent))

    # ── vectorized max-width for a band region ────────────────────────
    def _band_max_width(region):
        row_has    = region.any(axis=1)
        if not row_has.any():
            return 0
        left_cols  = np.argmax(region, axis=1)
        right_cols = w - 1 - np.argmax(region[:, ::-1], axis=1)
        widths     = np.where(row_has, right_cols - left_cols, 0)
        return int(widths.max())

    top_max_width    = _band_max_width(cropped[:band_h,  :])
    bottom_max_width = _band_max_width(cropped[-band_h:, :])

    # wider section at the bottom → need flip
    return bottom_max_width > top_max_width


def _flip_mask_180(mask):
    """
    Rotate the mask exactly 180 degrees in-place (equivalent to
    np.rot90 twice, but uses cv2 for consistency with the rest of
    the pipeline).
    """
    h, w   = mask.shape[:2]
    center = (w / 2.0, h / 2.0)
    M      = cv2.getRotationMatrix2D(center, 180.0, 1.0)
    return cv2.warpAffine(
        mask, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderValue=0
    )


def find_best_rotation(mask):
    """
    Rotate object vertically and ensure the wider / heavier end
    (e.g. bolt head) is always at the top.

    Pipeline
    --------
    1. Keep only the largest connected component (noise removal).
    2. PCA-based vertical alignment.
    3. Orientation check: compare maximum width in the top vs bottom
       quarter of the aligned mask.
    4. If the wider section is at the bottom, apply a 180-degree flip
       directly into best_mask — no flag is exposed to the caller.
    5. Recompute all geometry on the final mask.

    Returns
    -------
    (
        best_mask,       — final oriented binary mask (uint8, values 0/255)
        corrected_angle, — PCA correction angle in degrees
        pixel_height,    — object height in pixels
        (top_point,      — (x, y) at the topmost row of the final mask
         bottom_point),  — (x, y) at the bottommost row of the final mask
        center           — (cx, cy) centroid of the final mask
    )

    Orientation guarantee
    ---------------------
        top_point  -> wider / heavier section  (head of bolt)
        bottom_point -> tapered / narrower end  (tip of thread)
    """

    # Step 1 : remove noise blobs
    mask   = keep_largest_component(mask)
    mask   = (mask > 0).astype(np.uint8)

    center = get_mask_center(mask)

    if center is None:
        return (
            None,
            0,
            0,
            (None, None),
            None
        )

    # Step 2 : PCA-based vertical alignment
    angle           = get_orientation(mask)
    corrected_angle = angle - 90

    # normalise to (-90, +90]
    if corrected_angle < -90:
        corrected_angle += 180
    if corrected_angle > 90:
        corrected_angle -= 180

    best_mask = rotate_mask(mask, corrected_angle, center)

    # clean binary mask
    best_mask = (best_mask > 0).astype(np.uint8)
    best_mask = keep_largest_component(best_mask)
    best_mask = best_mask * 255

    upside_down=False

    # Step 3 & 4 : orientation check — flip baked into best_mask
    if _is_upside_down(best_mask):
        best_mask = _flip_mask_180(best_mask)
        best_mask = (best_mask > 0).astype(np.uint8)
        best_mask = keep_largest_component(best_mask)
        best_mask = best_mask * 255
        upside_down=True

    # Step 5 : recompute all geometry on the final mask
    center       = get_mask_center(best_mask)
    pixel_height = get_mask_height(best_mask)

    top_point, bottom_point = get_top_bottom_points(best_mask)

    return (
        best_mask,
        corrected_angle,
        pixel_height,
        (top_point, bottom_point),
        center,
        upside_down
    )
# =========================================================
# CREATE LARGEST CONTOUR MASK
# =========================================================

def create_largest_contour_mask(img):
    """
    Create filled white mask from largest contour.
    """

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV +
        cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None, None, None

    largest = max(
        contours,
        key=cv2.contourArea
    )

    mask = np.zeros(
        gray.shape,
        dtype=np.uint8
    )

    cv2.drawContours(
        mask,
        [largest],
        -1,
        255,
        thickness=-1
    )

    masked_result = cv2.bitwise_and(
        img,
        img,
        mask=mask
    )

    return (
        mask,
        masked_result,
        largest
    )


# ======================= WIDTH CALCULATIONS =========================================


def get_width_at_y(mask, y):

    h, w = mask.shape[:2]

    y = int(np.clip(y, 0, h - 1))

    row = mask[y]

    xs = np.where(row > 0)[0]

    if len(xs) == 0:
        return None

    left_x  = int(xs[0])
    right_x = int(xs[-1])

    width    = right_x - left_x
    center_x = (left_x + right_x) // 2

    return {
        "center": (center_x, y),
        "width":  width,
        "left":   (left_x,  y),
        "right":  (right_x, y)
    }


def find_max_width_near_point(mask, target_y, search_pixels):
    """
    SPEED: fully vectorized — no Python row loop.
    Slices the search window, computes all row widths at once with NumPy,
    then picks the widest row.
    """

    h, w = mask.shape[:2]

    y1 = max(0, target_y - search_pixels)
    y2 = min(h - 1, target_y + search_pixels) + 1   # +1 for slice end

    region = mask[y1:y2]                              # shape: (rows, w)

    row_has = region.any(axis=1)                      # bool per row

    if not row_has.any():
        return None

    # First nonzero column per row
    left_cols  = np.argmax(region, axis=1)
    # Last nonzero column per row (flip trick)
    right_cols = w - 1 - np.argmax(region[:, ::-1], axis=1)

    # Width = -1 for empty rows so argmax never picks them
    widths = np.where(row_has, right_cols - left_cols, -1)

    best_row  = int(np.argmax(widths))
    y         = y1 + best_row
    lx        = int(left_cols[best_row])
    rx        = int(right_cols[best_row])
    width_val = int(widths[best_row])

    return {
        "center": ((lx + rx) // 2, y),
        "width":  width_val,
        "left":   (lx, y),
        "right":  (rx, y)
    }


def find_top_center_bottom_max_widths(
    mask,
    top_bottom_points,
    center_point,
    move_percent=0.10
):

    (top_x, top_y), (bottom_x, bottom_y) = top_bottom_points

    total_height = abs(bottom_y - top_y)

    move_pixels = int(total_height * move_percent)

    top_target_y    = top_y    + move_pixels
    bottom_target_y = bottom_y - move_pixels
    center_target_y = center_point[1]

    search_pixels = move_pixels

    top_result = find_max_width_near_point(
        mask, top_target_y, search_pixels
    )

    center_result = find_max_width_near_point(
        mask, center_target_y, search_pixels
    )

    bottom_result = find_max_width_near_point(
        mask, bottom_target_y, search_pixels
    )

    return {
        "top":    top_result,
        "center": center_result,
        "bottom": bottom_result
    }


# ============================================================= BBOX


def get_rotated_box(
        bbox,
        angle_deg,
        top_bottom_point,
        size,
        expand_percent=10,
        text_offset=25):

    (xt, yt), (xb, yb) = top_bottom_point

    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    height = math.hypot(xb - xt, yb - yt)

    width = max(
        size["top"]["width"],
        size["center"]["width"],
        size["bottom"]["width"]
    )

    scale   = 1 + expand_percent / 100.0
    width  *= scale
    height *= scale

    rect = ((cx, cy), (width, height), angle_deg)

    box = cv2.boxPoints(rect)
    box = np.int32(box)

    p0, p1, p2, p3 = box.astype(np.float32)

    # Width label position
    width_mid = (p0 + p1) / 2
    vec = width_mid - np.array([cx, cy])
    vec /= np.linalg.norm(vec)
    width_text = (
        int(width_mid[0] + vec[0] * text_offset),
        int(width_mid[1] + vec[1] * text_offset)
    )

    # Height label position
    height_mid = (p1 + p2) / 2
    vec = height_mid - np.array([cx, cy])
    vec /= np.linalg.norm(vec)
    height_text = (
        int(height_mid[0] + vec[0] * text_offset),
        int(height_mid[1] + vec[1] * text_offset)
    )

    return box, width_text, height_text


# ==================================================

def get_rotated_line_from_bbox(bbox, angle_deg, length=50):

    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    rad = np.deg2rad(angle_deg + 90)

    dx = (length / 2) * np.cos(rad)
    dy = (length / 2) * np.sin(rad)

    pt1 = (int(cx - dx), int(cy - dy))
    pt2 = (int(cx + dx), int(cy + dy))

    return pt1, pt2


# =================== THREAD SIZE ========================


def get_thread_length(mask):
    """
    Returns:
        start_pt, end_pt, length_px

    SPEED: fully vectorized row-width computation — no Python for-loop.
    """

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask = (mask > 0).astype(np.uint8)

    h, w = mask.shape

    # --- Vectorized width profile ---
    row_has = mask.any(axis=1)                           # (h,) bool
    left_cols  = np.argmax(mask, axis=1)                 # first nonzero
    right_cols = w - 1 - np.argmax(mask[:, ::-1], axis=1)  # last nonzero
    widths = np.where(row_has, right_cols - left_cols, 0).astype(np.float32)

    valid = np.where(widths > 0)[0]

    if len(valid) == 0:
        return None

    # Smooth width profile
    smooth = cv2.GaussianBlur(
        widths.reshape(-1, 1),
        (1, 31),
        0
    ).flatten()

    # Widest row = head
    head_y = int(np.argmax(smooth))

    top_len    = head_y
    bottom_len = h - head_y

    if bottom_len > top_len:
        # head on top, thread below
        diff        = np.diff(smooth[head_y:])
        shoulder_y  = head_y + int(np.argmin(diff))
        tip_y       = int(valid[-1])
    else:
        # head on bottom, thread above
        diff        = np.diff(smooth[:head_y])
        shoulder_y  = int(np.argmin(diff))
        tip_y       = int(valid[0])

    # Centre x (vectorized)
    ys, xs = np.where(mask > 0)
    cx = int(np.mean(xs))

    length_px = abs(tip_y - shoulder_y)

    return (
        (cx, shoulder_y),
        (cx, tip_y),
        length_px
    )


# =========================================


class MaskAreaLock:

    def __init__(self,
                 learn_frames=5,
                 tolerance_percent=20):

        self.learn_frames       = learn_frames
        self.tolerance_percent  = tolerance_percent
        self.areas              = []
        self.reference_area     = None

    def update(self, mask):

        area = cv2.countNonZero(mask)

        # Learning stage
        if self.reference_area is None:

            self.areas.append(area)

            if len(self.areas) >= self.learn_frames:
                self.reference_area = np.mean(self.areas)
                print(f"Locked area = {self.reference_area:.0f}")

            return False

        # Compare stage
        diff_percent = (
            abs(area - self.reference_area)
            / self.reference_area
        ) * 100

        return diff_percent <= self.tolerance_percent
    


class BoltMeasurementFilter:
    def __init__(self, history_size=5, max_change_percent=0.15):
        self.history_size = history_size
        self.max_change_percent = max_change_percent

        self.history = {
            "fixed_bolt_height": deque(maxlen=history_size),
            "thread_height": deque(maxlen=history_size),
            "top_width": deque(maxlen=history_size),
            "center_width": deque(maxlen=history_size),
            "bottom_width": deque(maxlen=history_size),
        }

    def _filter_value(self, key, value):
        hist = self.history[key]

        if len(hist) < 3:
            hist.append(value)
            return value

        median = np.median(hist)

        # Reject sudden spike
        if median > 0:
            pct_change = abs(value - median) / median

            if pct_change > self.max_change_percent:
                value = int(round(median))

        hist.append(value)

        return int(round(np.median(hist)))

    def update(self, bolt_total_height, bolt_thread_height, result_width):

        top_width = result_width["top"]["width"]
        center_width = result_width["center"]["width"]
        bottom_width = result_width["bottom"]["width"]

        return {
            "fixed_bolt_height": self._filter_value(
                "fixed_bolt_height",
                bolt_total_height
            ),

            "thread_height": self._filter_value(
                "thread_height",
                bolt_thread_height
            ),

            "top_width": self._filter_value(
                "top_width",
                top_width
            ),

            "center_width": self._filter_value(
                "center_width",
                center_width
            ),

            "bottom_width": self._filter_value(
                "bottom_width",
                bottom_width
            ),
        }




#=========================rotate


# def rotate_contour_region(image, contour, angle):
#     h, w = image.shape[:2]

#     # Create white background
#     result = np.full_like(image, 255)

#     # Mask from contour
#     mask = np.zeros((h, w), dtype=np.uint8)
#     cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)

#     # Extract object
#     obj = cv2.bitwise_and(image, image, mask=mask)

#     # Rotation matrix
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)

#     # Rotate object and mask
#     obj_rot = cv2.warpAffine(
#         obj, M, (w, h),
#         flags=cv2.INTER_LINEAR,
#         borderValue=(0, 0, 0)
#     )

#     mask_rot = cv2.warpAffine(
#         mask, M, (w, h),
#         flags=cv2.INTER_NEAREST,
#         borderValue=0
#     )

#     # Paste rotated object onto white background
#     result[mask_rot > 0] = obj_rot[mask_rot > 0]

#     return result


def rotate_contour_region(image, contour, angle):
    h, w = image.shape[:2]

    pad = int(math.hypot(h, w) / 2) + 2

    image_pad = cv2.copyMakeBorder(
        image,
        pad, pad, pad, pad,
        cv2.BORDER_CONSTANT,
        value=(255,255,255)
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)

    mask_pad = cv2.copyMakeBorder(
        mask,
        pad, pad, pad, pad,
        cv2.BORDER_CONSTANT,
        value=0
    )

    ph, pw = image_pad.shape[:2]

    center = (pw // 2, ph // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    obj_rot = cv2.warpAffine(
        image_pad,
        M,
        (pw, ph),
        flags=cv2.INTER_LINEAR,
        borderValue=(255,255,255)
    )

    mask_rot = cv2.warpAffine(
        mask_pad,
        M,
        (pw, ph),
        flags=cv2.INTER_NEAREST,
        borderValue=0
    )

    result = np.full_like(obj_rot, 255)
    result[mask_rot > 0] = obj_rot[mask_rot > 0]

    return result