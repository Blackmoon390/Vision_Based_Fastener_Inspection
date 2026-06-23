import cv2
from ultralytics import YOLO
import torch
import texture_process_module_2 as tpm
import os

# =========================
# Create output folder
# =========================
save_folder = "saved_frames"
os.makedirs(save_folder, exist_ok=True)

# Frame counter
frame_id = 0

# Check GPU
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ─────────────────────────────────────────────
# SPEED SETTINGS — tune these for your use case
# ─────────────────────────────────────────────
PROCESS_EVERY_N = 3      # skip N-1 frames between inference runs (2–5 typical)
INFER_SCALE     = 0.5    # resize factor before YOLO (0.5 = half resolution)
SHOW_DEBUG_WINS = True   # set False in production to remove extra imshow cost

# Load custom YOLO model
model = YOLO(
    r"C:\python\computer_vision\Image-based-dimension-measurement-\model\yolo_bolt_nut_seg_best.pt"
)

locker = tpm.MaskAreaLock(
    learn_frames=5,
    tolerance_percent=10
)

# Video path
ved = r"C:\python\computer_vision\Image-based-dimension-measurement-\trail_works_phase2\20260601_112928.mp4"

# Open video
cap = cv2.VideoCapture(ved)

frame_count = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # ── SPEED: skip frames ──────────────────────────────────────────
    if frame_count % PROCESS_EVERY_N != 0:
        cv2.imshow("main", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue
    # ────────────────────────────────────────────────────────────────

    # ── SPEED: resize for inference, then map coords back ───────────
    h_orig, w_orig = frame.shape[:2]
    small = cv2.resize(frame, (0, 0), fx=INFER_SCALE, fy=INFER_SCALE)
    inv_scale = 1.0 / INFER_SCALE
    # ────────────────────────────────────────────────────────────────

    # Run inference on the smaller frame
    results = model(
        small,
        device=0,
        conf=0.5,
        verbose=False
    )

    for result in results:

        boxes = result.boxes

        for i, box in enumerate(boxes):

            # Scale coordinates back to original resolution
            x1, y1, x2, y2 = [int(v * inv_scale) for v in box.xyxy[0]]

            conf  = float(box.conf[0])
            print(conf)
            cls   = int(box.cls[0])
            label = model.names[cls]

            # Box width/height
            w = x2 - x1
            h = y2 - y1

            # Expand by 50%
            expand = 0.50
            pad_w  = int(w * expand / 2)
            pad_h  = int(h * expand / 2)

            # Expanded + clamped coordinates
            x1_new = max(0, x1 - pad_w)
            y1_new = max(0, y1 - pad_h)
            x2_new = min(w_orig, x2 + pad_w)
            y2_new = min(h_orig, y2 + pad_h)

            # Draw label on full-res frame
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1_new, y1_new - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Crop
            crop = frame[y1_new:y2_new, x1_new:x2_new]

            mask, masked_result, largest = tpm.create_largest_contour_mask(crop)

            matched = locker.update(mask)

            if matched:
                best_mask, angle, height, mask_top_bottom, center = \
                    tpm.find_best_rotation(mask)

                newbbox = (x1_new, y1_new, x2_new, y2_new)

                result_data = tpm.find_top_center_bottom_max_widths(
                    best_mask, mask_top_bottom, center, move_percent=0.10
                )

                thx, thy, thh = tpm.get_thread_length(best_mask)

                bbox_coor, widthcoor, heightcoor = tpm.get_rotated_box(
                    newbbox, angle, mask_top_bottom, result_data
                )

                

                cv2.putText(frame, f"{height} mm", widthcoor,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, f"{thh} mm",    heightcoor,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                vis = cv2.cvtColor(best_mask, cv2.COLOR_GRAY2BGR)
                cv2.line(vis, thx, thy, (0, 0, 255), 2)

                if bbox_coor is not None:
                    cv2.drawContours(frame, [bbox_coor], 0, (0, 255, 0), 2)

                for key in result_data:
                    data = result_data[key]
                    if data is None or data["center"] is None:
                        continue

                    cx_w, cy_w = data["center"]
                    left_pt    = data["left"]
                    right_pt   = data["right"]
                    width      = data["width"]

                    cv2.line(vis, left_pt, right_pt, (0, 255, 0), 2)
                    cv2.circle(vis, (cx_w, cy_w), 5, (0, 0, 255), -1)
                    cv2.putText(vis, f"{width} mm",  (cx_w + 10, cy_w),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    cv2.putText(vis, f"{height}",    (20, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    # save_path = os.path.join(
                    #     save_folder,
                    #     f"frame_{frame_id:05d}_obj_{i}.png"
                    # )
                    # cv2.imwrite(save_path, vis)

                frame_id += 1

                # ── SPEED: one combined window instead of 3 separate ones ──
                # frame = cv2.resize(frame,None,fx=0.3,fy=0.5,interpolation=cv2.INTER_LINEAR)
                cv2.imshow("main", frame)
                if SHOW_DEBUG_WINS:
                    cv2.imshow("mask_vis", vis)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

cap.release()
cv2.destroyAllWindows()