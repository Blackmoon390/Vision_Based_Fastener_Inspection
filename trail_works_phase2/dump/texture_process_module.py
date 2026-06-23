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

    gray = cv2.bilateralFilter(
        gray,
        9,
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
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
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
    Rotate mask without clipping
    """

    mask = (mask > 0).astype(np.uint8)

    h, w = mask.shape[:2]

    pad = max(h, w)

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


# def get_top_bottom_points(mask):
#     """
#     Get top and bottom points
#     """

#     contour = get_largest_contour(mask)

#     if contour is None:
#         return None, None

#     pts = contour.reshape(-1, 2)

#     ys = pts[:, 1]

#     min_y = ys.min()
#     max_y = ys.max()

#     top_pts = pts[ys == min_y]
#     bottom_pts = pts[ys == max_y]

#     top_x = int(top_pts[:, 0].mean())
#     bottom_x = int(bottom_pts[:, 0].mean())

#     top_point = (
#         top_x,
#         int(min_y)
#     )

#     bottom_point = (
#         bottom_x,
#         int(max_y)
#     )

#     return top_point, bottom_point

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
    Create filled white mask
    from largest contour
    """

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    blur = gray

    _, thresh = cv2.threshold(
        blur,
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


#======================= widht calualtions =========================================




def get_width_at_y(mask, y):

    h, w = mask.shape[:2]

    y = int(np.clip(y, 0, h - 1))

    row = mask[y]

    xs = np.where(row > 0)[0]

    if len(xs) == 0:
        return None

    left_x = int(xs.min())
    right_x = int(xs.max())

    width = right_x - left_x

    center_x = int((left_x + right_x) / 2)

    return {
        "center": (center_x, y),
        "width": width,
        "left": (left_x, y),
        "right": (right_x, y)
    }


def find_max_width_near_point(
    mask,
    target_y,
    search_pixels
):
    """
    Search nearby rows and return maximum width row
    """

    h = mask.shape[0]

    y1 = max(0, target_y - search_pixels)
    y2 = min(h - 1, target_y + search_pixels)

    best = None
    best_width = -1

    for y in range(y1, y2 + 1):

        data = get_width_at_y(mask, y)

        if data is None:
            continue

        if data["width"] > best_width:
            best_width = data["width"]
            best = data

    return best


def find_top_center_bottom_max_widths(
    mask,
    top_bottom_points,
    center_point,
    move_percent=0.10
):

    (top_x, top_y), (bottom_x, bottom_y) = top_bottom_points

    total_height = abs(bottom_y - top_y)

    # move 10%
    move_pixels = int(total_height * move_percent)

    # top moved inside
    top_target_y = top_y + move_pixels

    # bottom moved inside
    bottom_target_y = bottom_y - move_pixels

    center_target_y = center_point[1]

    # search around moved point
    search_pixels = move_pixels

    top_result = find_max_width_near_point(
        mask,
        top_target_y,
        search_pixels
    )

    center_result = find_max_width_near_point(
        mask,
        center_target_y,
        search_pixels
    )

    bottom_result = find_max_width_near_point(
        mask,
        bottom_target_y,
        search_pixels
    )

    return {
        "top": top_result,
        "center": center_result,
        "bottom": bottom_result
    }

#=============================================================bbox



# def get_rotated_box(bbox, angle_deg, top_bottom_point, size, expand_percent=10):

#     (xt, yt), (xb, yb) = top_bottom_point

#     x1, y1, x2, y2 = bbox
#     cx = (x1 + x2) / 2.0
#     cy = (y1 + y2) / 2.0

#     height = math.hypot(xb - xt, yb - yt)

#     width = max(
#         size["top"]["width"],
#         size["center"]["width"],
#         size["bottom"]["width"]
#     )

#     scale = 1 + expand_percent / 100.0

#     width *= scale
#     height *= scale

#     rect = (
#         (cx, cy),
#         (width, height),
#         angle_deg
#     )

#     box = cv2.boxPoints(rect)

#     return np.int32(box)



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

    scale = 1 + expand_percent / 100.0
    width *= scale
    height *= scale

    rect = ((cx, cy), (width, height), angle_deg)

    box = cv2.boxPoints(rect)
    box = np.int32(box)

    # box order:
    # p0----p1
    # |      |
    # p3----p2

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




#==================================================

def get_rotated_line_from_bbox(bbox, angle_deg, length=50):
    """
    Parameters
    ----------
    bbox : tuple
        (x1, y1, x2, y2)

    angle_deg : float
        Rotation angle in degrees

    length : int
        Line length in pixels

    Returns
    -------
    pt1, pt2 : tuple
        Rotated line endpoints
    """

    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # perpendicular to angle
    rad = np.deg2rad(angle_deg + 90)

    dx = (length / 2) * np.cos(rad)
    dy = (length / 2) * np.sin(rad)

    pt1 = (int(cx - dx), int(cy - dy))
    pt2 = (int(cx + dx), int(cy + dy))

    return pt1, pt2


#=================== thread size ========================

# def get_thread_length(mask):

#     if len(mask.shape) == 3:
#         mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

#     mask = (mask > 0).astype(np.uint8)

#     h, w = mask.shape

#     widths = []

#     for y in range(h):
#         xs = np.where(mask[y] > 0)[0]

#         if len(xs):
#             widths.append(xs[-1] - xs[0])
#         else:
#             widths.append(0)

#     widths = np.array(widths)

#     valid = np.where(widths > 0)[0]

#     if len(valid) == 0:
#         return None

#     # smooth widths
#     smooth = cv2.GaussianBlur(
#         widths.astype(np.float32).reshape(-1, 1),
#         (1, 31),
#         0
#     ).flatten()

#     # derivative
#     diff = np.diff(smooth)

#     # biggest negative drop = head -> thread transition
#     thread_start_y = np.argmin(diff)

#     # bottom of object
#     thread_end_y = valid[-1]

#     thread_length = thread_end_y - thread_start_y

#     return thread_start_y, thread_end_y, thread_length


def get_thread_length(mask):
    """
    Returns:
        start_pt
        end_pt
        length_px
    """

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask = (mask > 0).astype(np.uint8)

    h, w = mask.shape

    widths = np.zeros(h, dtype=np.int32)

    for y in range(h):
        xs = np.where(mask[y] > 0)[0]

        if len(xs):
            widths[y] = xs[-1] - xs[0]

    valid = np.where(widths > 0)[0]

    if len(valid) == 0:
        return None

    # smooth width profile
    smooth = cv2.GaussianBlur(
        widths.astype(np.float32).reshape(-1, 1),
        (1, 31),
        0
    ).flatten()

    # widest row belongs to head
    head_y = np.argmax(smooth)

    # determine which side is longer
    top_len = head_y
    bottom_len = h - head_y

    if bottom_len > top_len:
        # head on top, thread below

        diff = np.diff(smooth[head_y:])
        shoulder_y = head_y + np.argmin(diff)

        tip_y = valid[-1]

    else:
        # head on bottom, thread above

        diff = np.diff(smooth[:head_y])
        shoulder_y = np.argmin(diff)

        tip_y = valid[0]

    ys, xs = np.where(mask > 0)
    cx = int(np.mean(xs))

    length_px = abs(tip_y - shoulder_y)

    return (
        (cx, shoulder_y),
        (cx, tip_y),
        length_px
    )

#=========================================


class MaskAreaLock:

    def __init__(self,
                 learn_frames=5,
                 tolerance_percent=20):

        self.learn_frames = learn_frames
        self.tolerance_percent = tolerance_percent

        self.areas = []
        self.reference_area = None

    def update(self, mask):

        area = cv2.countNonZero(mask)

        # Learning stage
        if self.reference_area is None:

            self.areas.append(area)

            if len(self.areas) >= self.learn_frames:

                self.reference_area = np.mean(self.areas)

                print(
                    f"Locked area = {self.reference_area:.0f}"
                )

            return False

        # Compare stage
        diff_percent = (
            abs(area - self.reference_area)
            / self.reference_area
        ) * 100

        return diff_percent <= self.tolerance_percent