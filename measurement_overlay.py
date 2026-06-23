


import cv2


def draw_measurement_lines(
    img,
    maxtopbottom,
    resultwidth,
    thread_height_xy,
    measurements,
    color=(0, 255, 0),
    thickness=2,
):
    top = resultwidth["top"]
    bottom = resultwidth["bottom"]

    maxtop = maxtopbottom[0][1]
    maxbottom = maxtopbottom[1][1]

    # Offset factors
    move_percent_top = 0.90
    move_percent_bottom = 1.05

    # Vertical line positions
    x_right = int(top["center"][0] + top["width"] / 1.5)
    x_left = int(top["center"][0] - bottom["width"])
    x = x_right

    y_top = maxtop
    y_bottom = maxbottom

    # ==========================
    # Draw Measurement Lines
    # ==========================

    # Bolt Height (Vertical)
    cv2.line(
        img,
        (x, y_top),
        (x, y_bottom),
        color,
        thickness,
    )

    # Bolt Total Width (Top Horizontal)
    top_y = int(move_percent_top * y_top)

    cv2.line(
        img,
        (top["left"][0], top_y),
        (top["right"][0], top_y),
        (255, 0, 0),
        thickness,
    )

    # Bolt Center Width (Bottom Horizontal)
    bottom_y = int(move_percent_bottom * y_bottom)

    cv2.line(
        img,
        (bottom["left"][0], bottom_y),
        (bottom["right"][0], bottom_y),
        (255, 0, 0),
        thickness,
    )

    # Thread Height
    thread_x = int(x_left * move_percent_top)

    cv2.line(
        img,
        (thread_x, thread_height_xy[0][1]),
        (thread_x, thread_height_xy[1][1]),
        (255, 0, 255),
        thickness,
    )

    # ==========================
    # Draw Text
    # ==========================

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_thickness = 2

    # --------------------------
    # Bolt Height
    # --------------------------
    text = f"{measurements['bolt_height']:.2f} mm"

    mid_y = (y_top + y_bottom) // 2

    cv2.putText(
        img,
        text,
        (x + 10, mid_y),
        font,
        font_scale,
        color,
        text_thickness,
        cv2.LINE_AA,
    )

    # --------------------------
    # Thread Height
    # --------------------------
    text = f"{measurements['thread_height']:.2f} mm"

    (tw, th), _ = cv2.getTextSize(
        text,
        font,
        font_scale,
        text_thickness,
    )

    thread_mid_y = (thread_height_xy[0][1] + thread_height_xy[1][1]) // 2

    cv2.putText(
        img,
        text,
        (thread_x - tw - 10, thread_mid_y),
        font,
        font_scale,
        (255, 0, 255),
        text_thickness,
        cv2.LINE_AA,
    )

    # --------------------------
    # Bolt Total Width
    # --------------------------
    text = f"{measurements['bolt_total_width']:.2f} mm"

    (tw, th), _ = cv2.getTextSize(
        text,
        font,
        font_scale,
        text_thickness,
    )

    cv2.putText(
        img,
        text,
        (
            top["center"][0] - tw // 2,
            top_y - 10,
        ),
        font,
        font_scale,
        (255, 0, 0),
        text_thickness,
        cv2.LINE_AA,
    )

    # --------------------------
    # Bolt Center Width
    # --------------------------
    text = f"{measurements['bolt_center_width']:.2f} mm"

    (tw, th), _ = cv2.getTextSize(
        text,
        font,
        font_scale,
        text_thickness,
    )

    cv2.putText(
        img,
        text,
        (
            bottom["center"][0] - tw // 2,
            bottom_y + th + 10,
        ),
        font,
        font_scale,
        (255, 0, 0),
        text_thickness,
        cv2.LINE_AA,
    )

    return img