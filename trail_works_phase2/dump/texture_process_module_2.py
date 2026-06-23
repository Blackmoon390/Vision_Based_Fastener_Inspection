import cv2
import numpy as np
import math


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


def find_best_rotation(mask):
    """
    Rotate object vertically

    Returns:
    --------
    best_mask
    corrected_angle
    pixel_height
    (top_point, bottom_point)
    center
    """

    # remove noise blobs
    mask = keep_largest_component(mask)

    mask = (mask > 0).astype(np.uint8)

    center = get_mask_center(mask)

    if center is None:
        return (
            None,
            0,
            0,
            (None, None),
            None
        )

    # PCA orientation
    angle = get_orientation(mask)

    # rotate to vertical
    corrected_angle = angle - 90

    # normalize
    if corrected_angle < -90:
        corrected_angle += 180

    if corrected_angle > 90:
        corrected_angle -= 180

    # rotate
    best_mask = rotate_mask(
        mask,
        corrected_angle,
        center
    )

    # convert to white mask
    best_mask = (
        best_mask > 0
    ).astype(np.uint8)

    # remove rotation artifacts
    best_mask = keep_largest_component(
        best_mask
    )

    # final mask
    best_mask = best_mask * 255

    # recompute center after rotation
    center = get_mask_center(best_mask)

    # height
    pixel_height = get_mask_height(
        best_mask
    )

    # top/bottom
    top_point, bottom_point = \
        get_top_bottom_points(
            best_mask
        )

    return (
        best_mask,
        corrected_angle,
        pixel_height,
        (top_point, bottom_point),
        center
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