import cv2
from ultralytics import YOLO
import torch
import texture_process_module_3 as tpm
import os
import threading
import queue
import time
from collections import deque
import measurement_overlay

# =========================
# Create output folder
# =========================
save_folder = "saved_frames"
os.makedirs(save_folder, exist_ok=True)

# Check GPU
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ─────────────────────────────────────────────
# SPEED SETTINGS
# ─────────────────────────────────────────────
PROCESS_EVERY_N = 1      # skip N-1 frames between inference runs
INFER_SCALE     = 0.6    # resize factor before YOLO (0.5 = half resolution)
SHOW_DEBUG_WINS = True    # show mask_vis window
DISPLAY_SCALE_X = 0.3     # display resize X
DISPLAY_SCALE_Y = 0.5     # display resize Y

# Queue sizes — large enough to buffer bursts without stalling the reader
READER_QUEUE_SIZE    = 64  # raw frames from disk
PROCESS_QUEUE_SIZE   = 32  # processed/annotated frames ready for display
DISPLAY_QUEUE_SIZE   = 64  # display-ready frames

# ─────────────────────────────────────────────
# Load model + locker (done once in main thread)
# ─────────────────────────────────────────────
model = YOLO(
    r"C:\python\computer_vision\Image-based-dimension-measurement-\model\yolo_bolt_nut_seg_best.pt"
)
model.to(0 if torch.cuda.is_available() else "cpu")

# Warm-up pass so first real frame isn't slow
dummy = torch.zeros(1, 3, 320, 320).to(0 if torch.cuda.is_available() else "cpu")
_ = model(dummy, verbose=False)

locker = tpm.MaskAreaLock(learn_frames=5, tolerance_percent=10)

bolt_filter = tpm.BoltMeasurementFilter()
video_path = r"C:\python\computer_vision\Image-based-dimension-measurement-\trail_works_phase2\20260601_112928.mp4"
video_path=r"C:\python\computer_vision\Image-based-dimension-measurement-\trail_works_phase2\Nuts\20260612_174915.mp4"
# video_path=r"C:\Users\VISHNU\Downloads\20260527_134339.mp4"

# ─────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────
reader_queue  = queue.Queue(maxsize=READER_QUEUE_SIZE)
display_queue = queue.Queue(maxsize=DISPLAY_QUEUE_SIZE)

stop_event    = threading.Event()
frame_id      = 0            # shared across processing thread only
total_frames_read = 0

# FPS tracking (display thread)
fps_deque    = deque(maxlen=30)
start_time   = None


# ═══════════════════════════════════════════════════════════
# THREAD 1 — Frame Reader
#   Reads every frame from disk and puts (frame_number, frame)
#   into reader_queue.  Never drops frames.
# ═══════════════════════════════════════════════════════════
def reader_thread(cap):
    global total_frames_read
    fc = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        fc += 1
        # Block until there is space — guarantees no frame is lost
        reader_queue.put((fc, frame))
        total_frames_read = fc

    # Sentinel: tell processing thread we are done
    reader_queue.put(None)
    print(f"[Reader] Done — {fc} frames read.")


# ═══════════════════════════════════════════════════════════
# THREAD 2 — Inference + Annotation
#   Pulls from reader_queue, runs YOLO every PROCESS_EVERY_N
#   frames, annotates, and pushes (display_frame, vis_frame|None)
#   into display_queue.
# ═══════════════════════════════════════════════════════════
def processing_thread():
    global frame_id

    while not stop_event.is_set():
        item = reader_queue.get()
        if item is None:                      # sentinel
            display_queue.put(None)
            break

        fc, frame = item
        h_orig, w_orig = frame.shape[:2]
        vis_frame = None

        # ── Only run inference every N frames ────────────────────────
        if fc % PROCESS_EVERY_N == 0:
            small     = cv2.resize(frame, (0, 0), fx=INFER_SCALE, fy=INFER_SCALE)
            inv_scale = 1.0 / INFER_SCALE

            results = model(small, device=0 if torch.cuda.is_available() else "cpu",
                            conf=0.5, verbose=False)

            for result in results:
                for i, box in enumerate(result.boxes):

                    x1, y1, x2, y2 = [int(v * inv_scale) for v in box.xyxy[0]]
                    conf  = float(box.conf[0])
                    cls   = int(box.cls[0])
                    label = model.names[cls]

                    w = x2 - x1
                    h = y2 - y1

                    expand = 0.50
                    pad_w  = int(w * expand / 2)
                    pad_h  = int(h * expand / 2)

                    x1_new = max(0, x1 - pad_w)
                    y1_new = max(0, y1 - pad_h)
                    x2_new = min(w_orig, x2 + pad_w)
                    y2_new = min(h_orig, y2 + pad_h)

                    cv2.putText(frame, f"{label} {conf:.2f}",
                                (x1_new, y1_new - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    crop = frame[y1_new:y2_new, x1_new:x2_new]
                    if crop.size == 0:
                        continue

                    mask, colour_mask, contour_edge = tpm.create_largest_contour_mask(crop)

                    matched = locker.update(mask)

                    if matched:
                        best_mask, angle, height, mask_top_bottom, center,upside_down = \
                            tpm.find_best_rotation(mask)
                        roatedclr=tpm.rotate_contour_region(colour_mask,contour_edge,angle)
                        roatedclr=tpm._flip_mask_180(roatedclr)
                    

                        newbbox     = (x1_new, y1_new, x2_new, y2_new)
                        result_data = tpm.find_top_center_bottom_max_widths(
                            best_mask, mask_top_bottom, center, move_percent=0.10
                        )

                        print(mask_top_bottom)

                        print(best_mask.shape)
                        print(roatedclr.shape)
                        # roatedclr=measurement_overlay.draw_measurement_lines(roatedclr,mask_top_bottom,result_data,thh)

                        
                        
                        thx, thy, thh = tpm.get_thread_length(best_mask)

                        totre=bolt_filter.update(height,thh,result_data)
                        
                        
                        
                        data={"bolt height":totre["fixed_bolt_height"],
                              "bolt thread height":totre["thread_height"],
                              "bolt total width":totre["top_width"],
                              "bolt center width (thread)":totre["center_width"],
                              "bolt bottom width (thread)":totre["bottom_width"]
                              }

                              

                        bbox_coor, widthcoor, heightcoor = tpm.get_rotated_box(
                            newbbox, angle, mask_top_bottom, result_data
                        )

                        cv2.putText(frame, f"{height} mm", widthcoor,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        cv2.putText(frame, f"{thh} mm", heightcoor,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                        vis_frame = cv2.cvtColor(best_mask, cv2.COLOR_GRAY2BGR)
                        # cv2.line(vis_frame, thx, thy, (0, 0, 255), 2)

                        if bbox_coor is not None:
                            cv2.drawContours(frame, [bbox_coor], 0, (0, 255, 0), 2)

                        print(result_data)

                        # for key in result_data:
                        #     data = result_data[key]
                        #     if data is None or data["center"] is None:
                        #         continue

                        #     cx_w, cy_w = data["center"]
                        #     left_pt    = data["left"]
                        #     right_pt   = data["right"]
                        #     width      = data["width"]

                        #     cv2.line(vis_frame, left_pt, right_pt, (0, 255, 0), 2)
                        #     cv2.circle(vis_frame, (cx_w, cy_w), 5, (0, 0, 255), -1)
                        #     cv2.putText(vis_frame, f"{width} mm", (cx_w + 10, cy_w),
                        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        #     cv2.putText(vis_frame, f"{height}",   (20, 20),
                        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                        ms={'bolt_height': 29.49, 'thread_height': 25.18, 'bolt_total_width': 12.09, 'bolt_center_width': 5.87, 'bolt_bottom_width': 5.61, 'head_diameter': 12.09, 'thread_diameter': 5.87} 

                            
                        roatedclr=measurement_overlay.draw_measurement_lines(vis_frame,mask_top_bottom,result_data,(thx,thy),ms)

                        # Save vis frame
                        save_path = os.path.join(
                            save_folder,
                            f"frame_{frame_id:05d}_obj_{i}.png"
                        )
                        cv2.imwrite(save_path, vis_frame)

                        frame_id += 1

        # Resize for display
        display_frame = cv2.resize(
            frame, (0, 0),
            fx=DISPLAY_SCALE_X, fy=DISPLAY_SCALE_Y,
            interpolation=cv2.INTER_LINEAR
        )

        # Block until display queue has space
        display_queue.put((display_frame, vis_frame))

    print("[Processor] Done.")


# ═══════════════════════════════════════════════════════════
# MAIN THREAD — Display
#   Pulls from display_queue and calls imshow / waitKey.
#   Overlays FPS and prints total runtime at end.
# ═══════════════════════════════════════════════════════════
def main():
    global start_time

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Cannot open video:", video_path)
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay_ms = max(1, int(1000 / video_fps))   # target ms per display frame

    # Start background threads
    t_reader    = threading.Thread(target=reader_thread,    args=(cap,), daemon=True)
    t_processor = threading.Thread(target=processing_thread,              daemon=True)
    t_reader.start()
    t_processor.start()

    print(f"[Main] Video FPS: {video_fps:.1f} | target delay: {frame_delay_ms} ms")

    start_time     = time.perf_counter()
    displayed      = 0
    last_vis       = None

    while not stop_event.is_set():
        try:
            item = display_queue.get(timeout=5.0)
        except queue.Empty:
            print("[Main] Timed out waiting for frames.")
            break

        if item is None:    # sentinel — video finished
            break

        display_frame, vis_frame = item

        # FPS overlay
        now = time.perf_counter()
        fps_deque.append(now)
        if len(fps_deque) >= 2:
            elapsed_window = fps_deque[-1] - fps_deque[0]
            fps = (len(fps_deque) - 1) / elapsed_window if elapsed_window > 0 else 0.0
            cv2.putText(display_frame, f"FPS: {fps:.1f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("main", display_frame)

        if SHOW_DEBUG_WINS and vis_frame is not None:
            last_vis = vis_frame
        if SHOW_DEBUG_WINS and last_vis is not None:
            cv2.imshow("mask_vis", last_vis)

        displayed += 1

        # Pace display to match original video speed
        key = cv2.waitKey(frame_delay_ms) & 0xFF
        if key == ord("q"):
            stop_event.set()
            break

    # ── Cleanup ────────────────────────────────────────────
    total_time = time.perf_counter() - start_time
    stop_event.set()
    t_reader.join(timeout=3)
    t_processor.join(timeout=3)
    cap.release()
    cv2.destroyAllWindows()

    # ── Summary ────────────────────────────────────────────
    avg_fps = displayed / total_time if total_time > 0 else 0
    print("\n" + "=" * 45)
    print("          VIDEO PROCESSING COMPLETE")
    print("=" * 45)
    print(f"  Total frames read      : {total_frames_read}")
    print(f"  Total frames displayed : {displayed}")
    print(f"  Saved detection frames : {frame_id}")
    print(f"  Total runtime          : {total_time:.2f} s")
    print(f"  Average display FPS    : {avg_fps:.1f}")
    print(f"  Target video FPS       : {video_fps:.1f}")
    print("=" * 45)


if __name__ == "__main__":
    main()