import cv2
from ultralytics import YOLO
import torch
import texture_process_module as tpm
import os
import threading
import queue
import time
from collections import deque
from calibration import Calibration
from config import CALIBRATION
import config
from industrial_dashboard import render_dashboard, init_window, show_frame, set_window_scale, register_alert_callback
from bolt_lookup import lookup
import measurement_overlay
from report_generator import create_bolt_report


# =========================
# Output folder
# =========================
# save_folder = "saved_frames"
# os.makedirs(save_folder, exist_ok=True)

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ─────────────────────────────────────────────
# EMA Smoother — stabilises display values
# ─────────────────────────────────────────────
class MeasurementSmoother:
    """
    Per-field Exponential Moving Average smoother with outlier rejection.

    alpha  : EMA weight for new samples (0.0=frozen, 1.0=raw).
             Lower = smoother but slower to respond.
             Recommended: 0.15–0.25 for bolt measurements.

    max_delta_mm : if a new reading differs from the current EMA by more
                   than this amount it is treated as a spike and ignored.
                   Set to None to disable outlier rejection.
    """
    def __init__(self, alpha: float = 0.20, max_delta_mm: float = 3.0):
        self.alpha         = alpha
        self.max_delta_mm  = max_delta_mm
        self._state: dict  = {}           # field → current EMA value

    def update(self, new_vals: dict) -> dict:
        """
        Feed new_vals dict (field → float mm value).
        Returns dict with the smoothed values for all known fields.
        """
        for field, raw in new_vals.items():
            if field not in self._state:
                self._state[field] = raw   # first sample: accept as-is
            else:
                cur = self._state[field]
                # Outlier gate
                if self.max_delta_mm is not None and abs(raw - cur) > self.max_delta_mm:
                    continue               # spike — skip this sample
                self._state[field] = self.alpha * raw + (1.0 - self.alpha) * cur

        # Return rounded copies so the dashboard shows tidy 2-dp numbers
        return {f: round(v, 2) for f, v in self._state.items()}

    def reset(self):
        self._state.clear()

converter = Calibration(**CALIBRATION)


set_window_scale(1.0)




# ParameterEffectTune if...alpha=0.2020% new, 80% carry-over. Low = very stable, slow to updateValues too sluggish → increase to 0.3; still jittery → decrease to 0.1max_delta_mm=3.0Rejects any jump > 3mm in one frame as a spikeReal bolt swaps are missed → increase; still flickering → decrease to 1.5
# For a stationary bolt on a fixed camera, alpha=0.10 and max_delta_mm=2.0 will give nearly frozen display values. For a moving conveyor, use alpha=0.30 so it tracks faster. 


smoother  = MeasurementSmoother(alpha=0.20, max_delta_mm=3.0)  # tune alpha & max_delta_mm to taste


# ─────────────────────────────────────────────
# ALERT BUTTON TRIGGER
# ─────────────────────────────────────────────
_alert_trigger = threading.Event()   # set by button click, cleared after processing

def _on_alert_button_clicked() -> None:
    """Registered as the dashboard button callback.
    Sets the trigger flag so the processing thread can react."""
    _alert_trigger.set()
    # print("[Report Generated saved as PDF]")

register_alert_callback(_on_alert_button_clicked)


# ─────────────────────────────────────────────
# SPEED SETTINGS
# ─────────────────────────────────────────────
PROCESS_EVERY_N = 1      # run inference every N frames
INFER_SCALE     = 0.6    # YOLO input downscale

DASH_WIN = "BOLT INSPECTION SYSTEM"

READER_QUEUE_SIZE  = 64
DISPLAY_QUEUE_SIZE = 64


#for report genearation
num=0

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
model = YOLO(config.MODEL_PATH
)
model.to(0 if torch.cuda.is_available() else "cpu")

dummy = torch.zeros(1, 3, 320, 320).to(0 if torch.cuda.is_available() else "cpu")
_ = model(dummy, verbose=False)

locker      = tpm.MaskAreaLock(learn_frames=5, tolerance_percent=10)
bolt_filter = tpm.BoltMeasurementFilter()

BOLT_RESET_TIMEOUT = 5.0          # seconds before resetting filters
last_bolt_detected_time = None    # wall-clock time of last cls==1 detection

# video_path = r"C:\python\computer_vision\Image-based-dimension-measurement-\trail_works_phase2\20260601_112928.mp4"
# video_path = r"C:\python\computer_vision\Image-based-dimension-measurement-\trail_works_phase2\Nuts\20260612_174915.mp4"


# ─────────────────────────────────────────────
# Queues & shared state
# ─────────────────────────────────────────────
reader_queue  = queue.Queue(maxsize=READER_QUEUE_SIZE)
display_queue = queue.Queue(maxsize=DISPLAY_QUEUE_SIZE)

stop_event        = threading.Event()
frame_id          = 0
total_frames_read = 0

fps_deque  = deque(maxlen=30)
start_time = None

dash_lock  = threading.Lock()
dash_state = {
    # ── measurements (updated from mm_data on every detection) ──
    "bolt_height":       0.0,
    "thread_height":     0.0,
    "bolt_total_width":  0.0,
    "bolt_center_width": 0.0,
    "bolt_bottom_width": 0.0,
    "head_diameter":     0.0,   # not in px_data → stays 0 (no sensor for this)
    "thread_diameter":   0.0,   # same
    # ── masks ───────────────────────────────────────────────────
    "bolt_mask":        None,
    "orientation_mask": None,
    # ── system ──────────────────────────────────────────────────
    "fps":               0.0,
    "gpu_status":        "ACTIVE" if torch.cuda.is_available() else "CPU",
    "camera_status":     "ONLINE",
    "frame_id":          0,
    "inference_time_ms": 0.0,
    "status":            "--",
}

# ── mm_data key → dash_state key mapping ────────────────────
# calibration.px_to_mm() returns these fixed output keys (NOT the px_data input keys).
# Keys come directly from Calibration.px_to_mm() in calibration.py.
MM_KEY_MAP = {
    "bolt_height_mm":        "bolt_height",
    "bolt_thread_height_mm": "thread_height",
    "bolt_total_width_mm":   "bolt_total_width",
    "bolt_center_width_mm":  "bolt_center_width",
    "bolt_bottom_width_mm":  "bolt_bottom_width",
}


# ═══════════════════════════════════════════════════════════
# THREAD 1 — Frame Reader
# ═══════════════════════════════════════════════════════════
def reader_thread(cap):
    global total_frames_read
    fc = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        fc += 1
        reader_queue.put((fc, frame))
        total_frames_read = fc
    reader_queue.put(None)
    print(f"[Reader] Done — {fc} frames read.")


# ═══════════════════════════════════════════════════════════
# THREAD 2 — Inference + Annotation
# ═══════════════════════════════════════════════════════════
def processing_thread():
    global frame_id, last_bolt_detected_time, locker, bolt_filter,num

    while not stop_event.is_set():
        item = reader_queue.get()
        if item is None:
            display_queue.put(None)
            break

        fc, frame = item
        h_orig, w_orig = frame.shape[:2]
        vis_frame = None
        infer_ms  = 0.0

        if fc % PROCESS_EVERY_N == 0:
            t0    = time.perf_counter()
            small = cv2.resize(frame, (0, 0), fx=INFER_SCALE, fy=INFER_SCALE)
            inv   = 1.0 / INFER_SCALE

            results  = model(small,
                             device=0 if torch.cuda.is_available() else "cpu",
                             conf=0.5, verbose=False)
            infer_ms = (time.perf_counter() - t0) * 1000

            for result in results:
                for i, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = [int(v * inv) for v in box.xyxy[0]]
                    conf  = float(box.conf[0])
                    cls   = int(box.cls[0])
                    label = model.names[cls]

                    # ── Only process bolt detections (cls == 1) ──────────
                    if cls == 0:

                        # Bolt detected — stamp the time for reset watchdog
                        last_bolt_detected_time = time.perf_counter()

                        w = x2 - x1;  h = y2 - y1
                        expand = 0.50
                        pad_w  = int(w * expand / 2)
                        pad_h  = int(h * expand / 2)
                        x1n = max(0, x1 - pad_w);  y1n = max(0, y1 - pad_h)
                        x2n = min(w_orig, x2 + pad_w); y2n = min(h_orig, y2 + pad_h)
    
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1n, y1n - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
                        crop = frame[y1n:y2n, x1n:x2n]
                        if crop.size == 0:
                            continue
    
                        mask, colour_mask, contour_edge = tpm.create_largest_contour_mask(crop)
                        matched = locker.update(mask)
    
                        if matched:
                            best_mask, angle, height, mask_top_bottom, center,upside_down = \
                                tpm.find_best_rotation(mask)
                            
    
                            newbbox     = (x1n, y1n, x2n, y2n)
                            result_data = tpm.find_top_center_bottom_max_widths(
                                best_mask, mask_top_bottom, center, move_percent=0.10
                            )
                            
                            thx, thy, thh = tpm.get_thread_length(best_mask)
                            totre = bolt_filter.update(height, thh, result_data)
    
                            # ── Build px_data ──────────────────────────────────
                            # Input keys match exactly what Calibration.px_to_mm() expects.
                            px_data = {
                                "bolt height":                totre["fixed_bolt_height"],
                                "bolt thread height":         totre["thread_height"],
                                "bolt total width":           totre["top_width"],
                                "bolt center width (thread)": totre["center_width"],
                                "bolt bottom width (thread)": totre["bottom_width"],
                            }
                            mm_data = converter.px_to_mm(px_data)
                            # mm_data keys: bolt_height_mm, bolt_thread_height_mm,
                            #               bolt_total_width_mm, bolt_center_width_mm,
                            #               bolt_bottom_width_mm
    
                            # ── Map mm_data → dash_state (via EMA smoother) ─────
                            raw_vals = {}
                            for mm_key, dash_key in MM_KEY_MAP.items():
                                val = mm_data.get(mm_key)
                                if val is not None:
                                    raw_vals[dash_key] = float(val)
    
                            # head_diameter  = top_width  × width_scale  (bolt head span)
                            # thread_diameter = center_width × width_scale (thread shank)
                            raw_vals["head_diameter"]   = round(totre["top_width"]    * converter.width_scale, 2)
                            raw_vals["thread_diameter"] = round(totre["center_width"] * converter.width_scale, 2)
    
                            # Apply EMA smoothing + outlier rejection
                            new_vals = smoother.update(raw_vals)
                            # print("mm data:",mm_data)
                            # print(new_vals)
    
                            # ── Annotate frame ──────────────────────────────────
                            bbox_coor, widthcoor, heightcoor = tpm.get_rotated_box(
                                newbbox, angle, mask_top_bottom, result_data
                            )
                            cv2.putText(frame, f"{height} mm", widthcoor,
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                            cv2.putText(frame, f"{thh} mm", heightcoor,
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
                            vis_frame = cv2.cvtColor(best_mask, cv2.COLOR_GRAY2BGR)

                            # ── Alert button: save vis_frame on click ────────────
                            if _alert_trigger.is_set():
                                _alert_trigger.clear()
                                mm_frame=vis_frame.copy()
                                mm_frame=measurement_overlay.draw_measurement_lines(vis_frame,mask_top_bottom,result_data,(thx,thy),new_vals)
                                rotated_crop=tpm.rotate_contour_region(colour_mask,contour_edge,angle)
                                if upside_down:
                                    rotated_crop=tpm._flip_mask_180(rotated_crop)
                                bolt_head_height=mm_data['bolt_height_mm'] - mm_data['bolt_thread_height_mm']
                                    # print("thread height:",bolt_head_height)
                                ISO_data=lookup.find_bolt(mm_data['bolt_bottom_width_mm'],bolt_head_height,mm_data['bolt_total_width_mm'])
                                    # print(ISO_data)

                                # print("rotated_crop shape:",None if rotated_crop is None else rotated_crop.shape)
                                # print("mm_frame shape:",None if mm_frame is None else mm_frame.shape)
                                # print("ISO_data:", ISO_data)
                                
                                rp_id = f"BOLT-NO{num}_F_{frame_id}"
                                
                                create_bolt_report(mm_data,ISO_data,rp_id,rotated_crop,mm_frame,f"{rp_id}.pdf")
                        
                                num+=1
                                # save_path = os.path.join(
                                #     save_folder, f"frame_{frame_id:05d}_obj_{i}.png"
                                # )
                                # save_path2 = os.path.join(
                                #     save_folder, f"2frame_{frame_id:05d}_obj_{i}.png"
                                # )
                                # cv2.imwrite(save_path, mm_frame)
                                # cv2.imwrite(save_path2, rotated_crop)
                                frame_id += 1
                                # print(f"[Alert] vis_frame saved -> {save_path}")
                            


                            cv2.line(vis_frame, thx, thy, (0, 0, 255), 2)
    
                            if bbox_coor is not None:
                                cv2.drawContours(frame, [bbox_coor], 0, (0, 255, 0), 2)
    
                            for key in result_data:
                                d = result_data[key]
                                if d is None or d["center"] is None:
                                    continue
                                cx_w, cy_w = d["center"]
                                cv2.line(vis_frame, d["left"], d["right"], (0, 255, 0), 2)
                                cv2.circle(vis_frame, (cx_w, cy_w), 5, (0, 0, 255), -1)
                                cv2.putText(vis_frame, f"{d['width']} mm",
                                            (cx_w + 10, cy_w),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                                cv2.putText(vis_frame, f"{height}", (20, 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                                
    
                            # ── Update shared dash state ────────────────────────
                            with dash_lock:
                                dash_state.update(new_vals)
                                dash_state["bolt_mask"]        = mask
                                dash_state["orientation_mask"] = vis_frame
                                dash_state["frame_id"]         = frame_id
                                dash_state["inference_time_ms"] = infer_ms
                                # PASS/FAIL on bolt_height tolerance
                                bh = dash_state["bolt_height"]
                                dash_state["status"] = "PASS" if 25.0 <= bh <= 35.0 else "FAIL"

                            
    
            # ── No-bolt watchdog: reset filters after 5 s without a cls==1 hit ──
            if last_bolt_detected_time is not None:
                if (time.perf_counter() - last_bolt_detected_time) > BOLT_RESET_TIMEOUT:
                    locker      = tpm.MaskAreaLock(learn_frames=5, tolerance_percent=10)
                    bolt_filter = tpm.BoltMeasurementFilter()
                    smoother.reset()
                    last_bolt_detected_time = None   # arm for next bolt
                    print("[Watchdog] No bolt for 5 s — MaskAreaLock & BoltMeasurementFilter reset.")

        # Always update frame_id + infer_ms even on non-matched frames.
        # NOTE: measurement values are intentionally NOT reset here —
        # the panel holds the last valid detection until the next one arrives.
        with dash_lock:
            dash_state["frame_id"]          = frame_id
            dash_state["inference_time_ms"] = infer_ms
            dash_state["camera_status"]     = "ONLINE"

        # Push full-resolution annotated frame for display
        display_queue.put(frame)

    print("[Processor] Done.")


# ═══════════════════════════════════════════════════════════
# MAIN THREAD — single dashboard window, no other imshow
# ═══════════════════════════════════════════════════════════
def main():
    global start_time
    

    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    if not cap.isOpened():
        print("ERROR: Cannot open video:", config.VIDEO_SOURCE)
        return

    video_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay_ms = max(1, int(1000 / video_fps))

    t_reader    = threading.Thread(target=reader_thread,    args=(cap,), daemon=True)
    t_processor = threading.Thread(target=processing_thread,             daemon=True)
    t_reader.start()
    t_processor.start()

    init_window(DASH_WIN)     # ONE window, fullscreen

    print(f"[Main] Video FPS: {video_fps:.1f} | delay: {frame_delay_ms} ms")
    start_time = time.perf_counter()
    displayed  = 0

    while not stop_event.is_set():
        try:
            live_frame = display_queue.get(timeout=5.0)
        except queue.Empty:
            print("[Main] Timed out waiting for frames.")
            break

        if live_frame is None:
            break

        # ── FPS ──────────────────────────────────────────────
        now = time.perf_counter()
        fps_deque.append(now)
        fps = 0.0
        if len(fps_deque) >= 2:
            elapsed = fps_deque[-1] - fps_deque[0]
            fps = (len(fps_deque) - 1) / elapsed if elapsed > 0 else 0.0

        # ── Snapshot dash state ───────────────────────────────
        with dash_lock:
            snap = dict(dash_state)
        snap["fps"] = fps

        # ── Render: live_frame fills 1920×1080, panels on top ─
        # live_frame is the ORIGINAL full-resolution annotated frame.
        # dashboard scales it internally — no pre-scaling here.
        dashboard = render_dashboard(snap, live_frame=live_frame)

        # ── ONE imshow — the dashboard window only ────────────
        key = show_frame(dashboard, DASH_WIN)
        displayed += 1

        if key == ord("q"):
            stop_event.set()
            break

        # Pace to video FPS
        remaining = frame_delay_ms - int((time.perf_counter() - now) * 1000)
        if remaining > 1:
            cv2.waitKey(remaining)

    # ── Cleanup ───────────────────────────────────────────────
    total_time = time.perf_counter() - start_time
    stop_event.set()
    t_reader.join(timeout=3)
    t_processor.join(timeout=3)
    cap.release()
    cv2.destroyAllWindows()

    # avg_fps = displayed / total_time if total_time > 0 else 0
    # print("\n" + "=" * 45)
    # print("          VIDEO PROCESSING COMPLETE")
    # print("=" * 45)
    # print(f"  Total frames read      : {total_frames_read}")
    # print(f"  Total frames displayed : {displayed}")
    # print(f"  Saved detection frames : {frame_id}")
    # print(f"  Total runtime          : {total_time:.2f} s")
    # print(f"  Average display FPS    : {avg_fps:.1f}")
    # print(f"  Target video FPS       : {video_fps:.1f}")
    # print("=" * 45)


if __name__ == "__main__":
    main()
