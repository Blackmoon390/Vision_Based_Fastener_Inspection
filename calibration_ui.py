"""
Pixel Calibration Tool
=======================

A small Tkinter app that:
  1. Lets you pick an IMAGE or a VIDEO as the calibration source.
  2. If it's a video, grabs however many frames you choose (default 6),
     clustered around the start, middle, and end - instead of asking
     you to scrub through it manually.
  3. Runs your existing detection + contour/rotation measurement
     pipeline on each frame to get the object's HEIGHT and WIDTH in
     pixels.
  4. Lets you pick the best result row, type in the REAL height/width
     in millimeters, and save everything into CALIBRATION in config.py.

REQUIREMENTS / ASSUMPTIONS
---------------------------
- This file must live in the same folder as your existing `config.py`
  and `tpm.py` (the module with create_largest_contour_mask,
  find_best_rotation, and find_top_center_bottom_max_widths). Those
  functions are called exactly the way your original cal.py called
  them - I could not see tpm.py's internals, so if its signatures are
  different you'll need to adjust the calls inside `measure_frame()`.
- Python packages needed: opencv-python, ultralytics, torch.
  Pillow (PIL) is optional - only used to show a little image preview
  in the results table; the tool still works without it.
- config.MODEL_PATH must point to a valid YOLO .pt weights file.

Run with:  python calibration_ui.py
"""

import os
import re
import sys
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Make sure config.py / tpm.py next to this script are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import texture_process_module as tpm  # noqa: your existing project module


# ---------------------------------------------------------------------------
# Core pipeline helpers (cleaned up version of the logic in your cal.py)
# ---------------------------------------------------------------------------

def load_model():
    """Load the YOLO model pointed to by config.MODEL_PATH."""
    from ultralytics import YOLO
    return YOLO(config.MODEL_PATH)


def resize_for_inference(frame, max_dim=640):
    """Shrink a frame for faster inference; return (small_frame, inv_scale)
    so detected boxes can be scaled back up to the original frame size."""
    h, w = frame.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1:
        return frame, 1.0
    small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    inv = 1.0 / scale
    return small, inv


def measure_frame(frame, model):
    """Run detection on one frame and measure pixel height/width of every
    detected object. Returns a list of dicts sorted by confidence (desc):
        {"label", "conf", "bbox", "height_px", "width_px", "vis_frame"}
    "vis_frame" is the annotated mask visualization (best_mask with the
    top/center/bottom width lines, center dots, and px labels drawn on
    it) - this is what gets shown in the preview, not the raw frame.
    """
    import torch

    small, inv = resize_for_inference(frame)
    device = 0 if torch.cuda.is_available() else "cpu"
    results = model(small, device=device, conf=0.5, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v * inv) for v in box.xyxy[0]]
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            try:
                mask, colour_mask, contour_edge = tpm.create_largest_contour_mask(crop)
                best_mask, angle, height_px, mask_top_bottom, center, upside_down = \
                    tpm.find_best_rotation(mask)
                result_data = tpm.find_top_center_bottom_max_widths(
                    best_mask, mask_top_bottom, center, move_percent=0.10
                )
                width_px = result_data["top"]["width"]

                vis_frame = cv2.cvtColor(best_mask, cv2.COLOR_GRAY2BGR)
                cv2.line(vis_frame, mask_top_bottom[0], mask_top_bottom[1], (0, 0, 255), 2)

                for key in result_data:
                    d = result_data[key]
                    if d is None or d["center"] is None:
                        continue
                    cx_w, cy_w = d["center"]
                    cv2.line(vis_frame, d["left"], d["right"], (0, 255, 0), 2)
                    cv2.circle(vis_frame, (cx_w, cy_w), 5, (0, 0, 255), -1)
                    cv2.putText(vis_frame, f"{width_px} px",
                                (cx_w + 10, cy_w),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    cv2.putText(vis_frame, f"{height_px} px", (20, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            except Exception:
                # measurement failed for this particular box - skip it
                continue

            detections.append({
                "label": label,
                "conf": conf,
                "bbox": (x1, y1, x2, y2),
                "height_px": float(height_px),
                "width_px": float(width_px),
                "vis_frame": vis_frame,
            })

    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections


def extract_sample_frames(video_path, num_frames=6):
    """Grab `num_frames` frames from a video, clustered around the start,
    middle, and end (roughly a third of the frames from each region)."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError("Could not determine frame count for this video.")

    num_frames = max(1, int(num_frames))
    if num_frames >= total:
        indices = list(range(total))
    else:
        base = num_frames // 3
        rem = num_frames % 3
        sizes = [base, base, base]
        for i in range(rem):
            sizes[i] += 1  # give any leftover frames to the start cluster(s) first

        mid = total // 2
        starts = [0, max(0, mid - sizes[1] // 2), max(0, total - sizes[2])]

        raw_indices = []
        for start, size in zip(starts, sizes):
            for offset in range(size):
                raw_indices.append(start + offset)

        indices = sorted(set(max(0, min(total - 1, i)) for i in raw_indices))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append((idx, frame))
    cap.release()
    return frames



def save_calibration_to_config(config_path, height_mm, height_px, width_mm, width_px):
    """Rewrite the CALIBRATION dict inside config.py, leaving everything
    else in the file (MODEL_PATH, VIDEO_SOURCE, etc.) untouched."""
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    new_block = (
        "CALIBRATION = {\n"
        f'    "ref_height_mm": {height_mm},\n'
        f'    "ref_height_px": {height_px},\n'
        f'    "ref_width_mm": {width_mm},\n'
        f'    "ref_width_px": {width_px}\n'
        "}"
    )

    pattern = re.compile(r"CALIBRATION\s*=\s*\{.*?\}", re.DOTALL)
    if pattern.search(content):
        content = pattern.sub(new_block, content, count=1)
    else:
        content = new_block + "\n\n" + content

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Tkinter UI
# ---------------------------------------------------------------------------

class CalibrationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pixel Calibration Tool")
        self.geometry("780x640")

        self.model = None
        self.model_load_failed = None
        self.source_path = None
        self.detections = []  # list of dicts, one per detected object/frame
        self.work_queue = queue.Queue()
        self._preview_imgtk = None  # keep a reference so Tk doesn't GC it

        self._build_ui()
        self._prefill_from_config()
        self.after(100, self._poll_queue)

        # Load the YOLO model in the background right away, using
        # config.MODEL_PATH, so it's ready before you hit Run Detection.
        threading.Thread(target=self._preload_model, daemon=True).start()

    # ---------- UI construction ----------
    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        # 0. Model info (from config.py)
        model_frame = ttk.Frame(self)
        model_frame.pack(fill="x", **pad)
        ttk.Label(model_frame, text="YOLO model (config.MODEL_PATH):").pack(side="left")
        ttk.Label(model_frame, text=config.MODEL_PATH, foreground="blue").pack(side="left", padx=6)
        self.model_status_label = ttk.Label(model_frame, text="loading...", foreground="orange")
        self.model_status_label.pack(side="left", padx=10)

        # 1. Source selection
        src_frame = ttk.LabelFrame(self, text="1. Select source")
        src_frame.pack(fill="x", **pad)

        self.source_var = tk.StringVar(value="image")
        ttk.Radiobutton(src_frame, text="Image", variable=self.source_var, value="image").pack(side="left", padx=8)
        ttk.Radiobutton(src_frame, text="Video", variable=self.source_var, value="video").pack(side="left", padx=8)
        ttk.Button(src_frame, text="Browse...", command=self.browse_file).pack(side="left", padx=8)
        self.path_label = ttk.Label(src_frame, text="No file selected", foreground="gray")
        self.path_label.pack(side="left", padx=8)

        ttk.Label(src_frame, text="Frames to sample (video only):").pack(side="left", padx=(20, 2))
        self.num_frames_var = tk.IntVar(value=6)
        ttk.Spinbox(src_frame, from_=1, to=50, width=4, textvariable=self.num_frames_var).pack(side="left")

        # Run button + status
        run_frame = ttk.Frame(self)
        run_frame.pack(fill="x", **pad)
        self.run_btn = ttk.Button(run_frame, text="Run Detection", command=self.run_detection)
        self.run_btn.pack(side="left")
        self.status_label = ttk.Label(run_frame, text="")
        self.status_label.pack(side="left", padx=10)

        # 2. Results table
        results_frame = ttk.LabelFrame(self, text="2. Detection results (click a row to use it)")
        results_frame.pack(fill="both", expand=True, **pad)

        cols = ("frame", "label", "conf", "height_px", "width_px")
        self.tree = ttk.Treeview(results_frame, columns=cols, show="headings", height=8)
        headings = {"frame": "Frame #", "label": "Label", "conf": "Conf",
                    "height_px": "Height (px)", "width_px": "Width (px)"}
        for c in cols:
            self.tree.heading(c, text=headings[c])
            self.tree.column(c, width=120, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_select_row)

        scroll = ttk.Scrollbar(results_frame, orient="vertical", command=self.tree.yview)
        scroll.pack(side="left", fill="y")
        self.tree.configure(yscrollcommand=scroll.set)

        self.preview_label = ttk.Label(results_frame)
        self.preview_label.pack(side="left", padx=10)

        # 3. Selected measurement
        sel_frame = ttk.LabelFrame(self, text="3. Selected pixel measurement (auto-filled or edit manually)")
        sel_frame.pack(fill="x", **pad)
        ttk.Label(sel_frame, text="Height (px):").grid(row=0, column=0, padx=6, pady=4, sticky="e")
        self.sel_height_entry = ttk.Entry(sel_frame, width=12)
        self.sel_height_entry.grid(row=0, column=1, padx=6, pady=4, sticky="w")
        ttk.Label(sel_frame, text="Width (px):").grid(row=0, column=2, padx=6, pady=4, sticky="e")
        self.sel_width_entry = ttk.Entry(sel_frame, width=12)
        self.sel_width_entry.grid(row=0, column=3, padx=6, pady=4, sticky="w")

        # 4. Manual mm entry
        mm_frame = ttk.LabelFrame(self, text="4. Enter known real-world size (mm)")
        mm_frame.pack(fill="x", **pad)
        ttk.Label(mm_frame, text="Height (mm):").grid(row=0, column=0, padx=6, pady=6, sticky="e")
        self.height_mm_entry = ttk.Entry(mm_frame, width=12)
        self.height_mm_entry.grid(row=0, column=1, padx=6, pady=6, sticky="w")
        ttk.Label(mm_frame, text="Width (mm):").grid(row=0, column=2, padx=6, pady=6, sticky="e")
        self.width_mm_entry = ttk.Entry(mm_frame, width=12)
        self.width_mm_entry.grid(row=0, column=3, padx=6, pady=6, sticky="w")

        # 5. Save
        save_frame = ttk.Frame(self)
        save_frame.pack(fill="x", **pad)
        ttk.Button(save_frame, text="Save to config.py", command=self.save_calibration).pack(side="left")

    # ---------- actions ----------
    def _prefill_from_config(self):
        """Pre-fill all fields from current config.py values."""
        # Video source
        video_source = getattr(config, "VIDEO_SOURCE", None)
        if video_source and os.path.exists(video_source):
            self.source_var.set("video")
            self.source_path = video_source
            self.path_label.config(text=os.path.basename(video_source), foreground="black")

        # Calibration values — fill all 4 entry fields from config
        cal = getattr(config, "CALIBRATION", {})
        if cal.get("ref_height_px"):
            self.sel_height_entry.delete(0, tk.END)
            self.sel_height_entry.insert(0, str(cal["ref_height_px"]))
        if cal.get("ref_width_px"):
            self.sel_width_entry.delete(0, tk.END)
            self.sel_width_entry.insert(0, str(cal["ref_width_px"]))
        if cal.get("ref_height_mm"):
            self.height_mm_entry.delete(0, tk.END)
            self.height_mm_entry.insert(0, str(cal["ref_height_mm"]))
        if cal.get("ref_width_mm"):
            self.width_mm_entry.delete(0, tk.END)
            self.width_mm_entry.insert(0, str(cal["ref_width_mm"]))

    def _preload_model(self):
        try:
            self.model = load_model()
            self.work_queue.put(("model_ready", None))
        except Exception as e:
            self.model_load_failed = str(e)
            self.work_queue.put(("model_error", str(e)))

    def browse_file(self):
        if self.source_var.get() == "image":
            path = filedialog.askopenfilename(
                title="Select calibration image",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
            )
        else:
            path = filedialog.askopenfilename(
                title="Select calibration video",
                filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
            )
        if path:
            self.source_path = path
            self.path_label.config(text=os.path.basename(path), foreground="black")

    def run_detection(self):
        if not self.source_path:
            messagebox.showwarning("No file", "Please select an image or video file first.")
            return
        self.run_btn.config(state="disabled")
        self.status_label.config(text="Working...")
        self.tree.delete(*self.tree.get_children())
        self.detections = []
        threading.Thread(target=self._run_detection_worker, daemon=True).start()

    def _run_detection_worker(self):
        try:
            if self.model is None:
                self.work_queue.put(("status", "Model still loading, please wait..."))
                while self.model is None:
                    if self.model_load_failed:
                        raise RuntimeError(f"Model failed to load: {self.model_load_failed}")
                    time.sleep(0.2)

            self.work_queue.put(("status", "Reading frame(s)..."))
            if self.source_var.get() == "image":
                frame = cv2.imread(self.source_path)
                if frame is None:
                    raise RuntimeError("Could not read the selected image.")
                frames = [(0, frame)]
            else:
                frames = extract_sample_frames(self.source_path, num_frames=self.num_frames_var.get())
                if not frames:
                    raise RuntimeError("Could not extract frames from the selected video.")

            self.work_queue.put(("status", f"Running detection on {len(frames)} frame(s)..."))
            all_results = []
            for idx, frame in frames:
                for d in measure_frame(frame, self.model):
                    d["frame_idx"] = idx
                    all_results.append(d)

            self.work_queue.put(("results", all_results))
        except Exception as e:
            self.work_queue.put(("error", str(e)))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.work_queue.get_nowait()
                if kind == "status":
                    self.status_label.config(text=payload)
                elif kind == "model_ready":
                    self.model_status_label.config(text="model loaded", foreground="green")
                elif kind == "model_error":
                    self.model_status_label.config(text="model failed to load", foreground="red")
                    messagebox.showerror("Model load error", payload)
                elif kind == "results":
                    self._populate_results(payload)
                    self.status_label.config(text=f"Done. {len(payload)} detection(s) found.")
                    self.run_btn.config(state="normal")
                elif kind == "error":
                    messagebox.showerror("Error", payload)
                    self.status_label.config(text="Error.")
                    self.run_btn.config(state="normal")
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _populate_results(self, results):
        self.detections = results
        if not results:
            messagebox.showinfo("No detections", "No objects were detected in the sampled frame(s).")
            return
        for i, d in enumerate(results):
            self.tree.insert("", "end", iid=str(i), values=(
                d["frame_idx"], d["label"], f'{d["conf"]:.2f}',
                f'{d["height_px"]:.1f}', f'{d["width_px"]:.1f}'
            ))

    def on_select_row(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        i = int(sel[0])
        d = self.detections[i]
        self.sel_height_entry.delete(0, tk.END)
        self.sel_height_entry.insert(0, f'{d["height_px"]:.2f}')
        self.sel_width_entry.delete(0, tk.END)
        self.sel_width_entry.insert(0, f'{d["width_px"]:.2f}')
        self._show_preview(d)

    def _show_preview(self, d):
        if not HAS_PIL:
            return
        vis_frame = d.get("vis_frame")
        if vis_frame is None:
            return
        frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        scale = 260 / max(h, w)
        frame_rgb = cv2.resize(frame_rgb, (max(1, int(w * scale)), max(1, int(h * scale))))
        img = Image.fromarray(frame_rgb)
        self._preview_imgtk = ImageTk.PhotoImage(img)
        self.preview_label.config(image=self._preview_imgtk)

    def save_calibration(self):
        try:
            height_px = float(self.sel_height_entry.get())
            width_px = float(self.sel_width_entry.get())
        except ValueError:
            messagebox.showwarning("No pixel values", "Enter numeric pixel values for height and width (or click a result row above).")
            return
        try:
            height_mm = float(self.height_mm_entry.get())
            width_mm = float(self.width_mm_entry.get())
        except ValueError:
            messagebox.showwarning("Invalid input", "Enter numeric mm values for height and width.")
            return
        config_path = os.path.abspath(config.__file__).replace(".pyc", ".py")
        print(f"[SAVE] Writing to: {config_path}")
        print(f"[SAVE] height_mm={height_mm} height_px={height_px} width_mm={width_mm} width_px={width_px}")
        try:
            save_calibration_to_config(config_path, height_mm, height_px, width_mm, width_px)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            return
        import importlib
        importlib.reload(config)
        messagebox.showinfo(
            "Saved",
            f"Calibration saved to {config_path}\n\n"
            f"ref_height_mm={height_mm}  ref_height_px={height_px}\n"
            f"ref_width_mm={width_mm}   ref_width_px={width_px}"
        )


if __name__ == "__main__":
    CalibrationApp().mainloop()
