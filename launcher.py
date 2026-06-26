"""
launcher.py – Vision-Based Automated Fastener Inspection
=========================================================
Flow:
    Splash  →  Home Menu  →  [Calibration | Configuration | Run Application]

Pages
-----
  Home          : Three large nav buttons
  Calibration   : Launches cal.py via subprocess, shows live status
  Configuration : Edit model path and video source → saves config.py
  Run           : Validates then launches main.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import threading
import sys
import os
import re
import string
import importlib.util
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Names to search for – edit these if your folder/file names change
# ─────────────────────────────────────────────────────────────────────────────
# The "Fix Paths" button searches every drive for a folder called
# PROJECT_FOLDER_NAME containing MAIN_SCRIPT_NAME and CALIBRATION_SCRIPT_NAME,
# then stores whatever it finds in the resolved-path variables below.
PROJECT_FOLDER_NAME     = "Vision_Based_Fastener_Inspection"
MAIN_SCRIPT_NAME        = "main.py"
CALIBRATION_SCRIPT_NAME = "calibration_ui.py"
CONFIG_SCRIPT_NAME      = "config.py"

# ─────────────────────────────────────────────────────────────────────────────
# Resolved paths – filled in automatically by the "Fix Paths" button
# ─────────────────────────────────────────────────────────────────────────────
# Empty until "Fix Paths" is run at least once. Do not need to be edited by
# hand; the button rewrites these three lines in this file after a search.
PROJECT_FOLDER = ""
MAIN_SCRIPT_PATH = ""
CALIBRATION_SCRIPT_PATH = ""
CONFIG_SCRIPT_PATH = ""

# ─────────────────────────────────────────────────────────────────────────────
# Where things live on disk, whether running as launcher.py or as a
# PyInstaller --onefile exe (which extracts to a temp _MEIxxxxxx folder that
# is wiped on exit, so we must NOT write/read next to sys.executable's
# extraction point — we use the real exe/script folder instead).
# ─────────────────────────────────────────────────────────────────────────────

def _app_dir() -> str:
    """
    Folder that should be treated as 'next to the launcher':
      - Frozen exe (PyInstaller): folder containing the .exe itself
        (sys.executable), NOT the temp _MEIxxxxxx extraction folder.
      - Plain script: folder containing this .py file.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


PATHS_CONFIG_FILE = os.path.join(_app_dir(), "launcher_paths.json")


def _default_script_paths():
    """Fallback to files sitting next to the launcher/exe (old behaviour)."""
    here = _app_dir()
    return (os.path.join(here, MAIN_SCRIPT_NAME),
            os.path.join(here, CALIBRATION_SCRIPT_NAME),
            os.path.join(here, CONFIG_SCRIPT_NAME))


def load_saved_paths():
    """Load previously-found paths from launcher_paths.json, if it exists."""
    global PROJECT_FOLDER, MAIN_SCRIPT_PATH, CALIBRATION_SCRIPT_PATH, CONFIG_SCRIPT_PATH
    if not os.path.exists(PATHS_CONFIG_FILE):
        return
    try:
        import json
        with open(PATHS_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        PROJECT_FOLDER = data.get("PROJECT_FOLDER", "")
        MAIN_SCRIPT_PATH = data.get("MAIN_SCRIPT_PATH", "")
        CALIBRATION_SCRIPT_PATH = data.get("CALIBRATION_SCRIPT_PATH", "")
        CONFIG_SCRIPT_PATH = data.get("CONFIG_SCRIPT_PATH", "")
    except Exception:
        pass  # Corrupt or unreadable config -> just fall back to search/defaults


def get_main_path() -> str:
    """Resolve main.py path: saved path wins, else same-folder fallback."""
    if MAIN_SCRIPT_PATH and os.path.exists(MAIN_SCRIPT_PATH):
        return MAIN_SCRIPT_PATH
    return _default_script_paths()[0]


def get_calibration_path() -> str:
    """Resolve calibration_ui.py path: saved path wins, else fallback."""
    if CALIBRATION_SCRIPT_PATH and os.path.exists(CALIBRATION_SCRIPT_PATH):
        return CALIBRATION_SCRIPT_PATH
    return _default_script_paths()[1]


def get_config_path() -> str:
    """Resolve config.py path: saved path wins, else same-folder fallback."""
    if CONFIG_SCRIPT_PATH and os.path.exists(CONFIG_SCRIPT_PATH):
        return CONFIG_SCRIPT_PATH
    return _default_script_paths()[2]


def find_project_paths():
    """
    Search every local drive for the project folder, then for the two
    scripts. Returns (project_folder, main_path, calibration_path, config_path) as
    Path objects, with any not found left as None.
    """
    project_folder = None
    main_path = None
    calibration_path = None
    config_path = None

    drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]

    # ---------- Step 1: Find project folder ----------
    for drive in drives:
        try:
            for folder in Path(drive).rglob(PROJECT_FOLDER_NAME):
                if folder.is_dir():
                    calibration = folder / CALIBRATION_SCRIPT_NAME
                    main = folder / MAIN_SCRIPT_NAME
                    config = folder / CONFIG_SCRIPT_NAME
                    if calibration.is_file() and main.is_file():
                        project_folder = folder
                        calibration_path = calibration
                        main_path = main
                        config_path = config if config.is_file() else None
                        break
            if project_folder:
                break
        except (PermissionError, OSError):
            continue

    # ---------- Step 2: Fallback search ----------
    if project_folder is None:
        for drive in drives:
            try:
                for calibration in Path(drive).rglob(CALIBRATION_SCRIPT_NAME):
                    main = calibration.parent / MAIN_SCRIPT_NAME
                    if main.is_file():
                        project_folder = calibration.parent
                        calibration_path = calibration
                        main_path = main
                        config = calibration.parent / CONFIG_SCRIPT_NAME
                        config_path = config if config.is_file() else None
                        break
                if project_folder:
                    break
            except (PermissionError, OSError):
                continue

    return project_folder, main_path, calibration_path, config_path


def persist_paths_to_file(project_folder, main_path, calibration_path, config_path=None):
    """
    Save the found paths to launcher_paths.json next to the launcher/exe so
    they survive a restart. (Rewriting our own source doesn't work once
    frozen into a PyInstaller exe, since there's no editable .py on disk
    at runtime — a JSON file next to the exe is used instead.)
    """
    import json
    data = {
        "PROJECT_FOLDER": str(project_folder) if project_folder else "",
        "MAIN_SCRIPT_PATH": str(main_path) if main_path else "",
        "CALIBRATION_SCRIPT_PATH": str(calibration_path) if calibration_path else "",
        "CONFIG_SCRIPT_PATH": str(config_path) if config_path else "",
    }
    with open(PATHS_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# Load any previously-saved paths immediately at import time.
load_saved_paths()


def get_python_launch_cmd(script_path: str) -> list:
    """
    Build the subprocess command to run a .py script without popping up a
    console/CMD window.

    When running as a plain script, sys.executable is the Python
    interpreter, so [sys.executable, script_path] is correct (and since the
    launcher itself was started without a console, the child inherits that).

    When frozen into a PyInstaller exe, sys.executable is this very exe
    (there is no bundled Python interpreter to call), so calling
    [sys.executable, script_path] would just re-run this exe. We fall back
    to "pythonw" on PATH instead — the windowless Python interpreter, which
    runs scripts without ever opening a console window. If pythonw isn't
    found, "python" is used as a last resort (paired with
    CREATE_NO_WINDOW in the Popen call so it still won't show a window).
    """
    if getattr(sys, "frozen", False):
        import shutil as _shutil
        interpreter = _shutil.which("pythonw") or "python"
        return [interpreter, script_path]
    return [sys.executable, script_path]


# On Windows, this flag stops a child process from opening its own console
# window. Used for every subprocess launch below.
NO_WINDOW_FLAGS = 0
if sys.platform == "win32":
    NO_WINDOW_FLAGS = subprocess.CREATE_NO_WINDOW



# ─────────────────────────────────────────────────────────────────────────────
# Design tokens – dark industrial theme
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":           "#12151C",
    "surface":      "#1A1E28",
    "card":         "#1F2433",
    "border":       "#2A2F40",
    "accent":       "#2979FF",
    "accent_h":     "#1A54C4",
    "success":      "#00C853",
    "success_h":    "#009624",
    "warning":      "#FF8F00",
    "warning_h":    "#E65100",
    "danger":       "#D32F2F",
    "fg":           "#E8EAF0",
    "fg2":          "#8A93AB",
    "fg3":          "#454D61",
    "entry":        "#0D1018",
    "header":       "#0D1018",
    "splash":       "#0A0C12",
}

TITLE_FONT  = ("Segoe UI", 26, "bold")
HEAD_FONT   = ("Segoe UI", 13, "bold")
LABEL_FONT  = ("Segoe UI", 10)
SMALL_FONT  = ("Segoe UI", 9)
MONO_FONT   = ("Consolas", 10)
BTN_FONT    = ("Segoe UI", 11, "bold")
NAV_FONT    = ("Segoe UI", 14, "bold")
NAV_SUB     = ("Segoe UI", 9)

# ─────────────────────────────────────────────────────────────────────────────
# Config I/O
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CALIB = {
    "ref_height_mm": 30,
    "ref_height_px": 355,
    "ref_width_mm":  12,
    "ref_width_px":  137,
}


def load_config() -> dict:
    """Dynamically load config.py and return values."""
    path = get_config_path()
    if not os.path.exists(path):
        return {**DEFAULT_CALIB, "model_path": "", "video_source": 0}
    spec = importlib.util.spec_from_file_location("_cfg", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calib = getattr(mod, "CALIBRATION", DEFAULT_CALIB)
    return {
        "ref_height_mm": calib.get("ref_height_mm", DEFAULT_CALIB["ref_height_mm"]),
        "ref_height_px": calib.get("ref_height_px", DEFAULT_CALIB["ref_height_px"]),
        "ref_width_mm":  calib.get("ref_width_mm",  DEFAULT_CALIB["ref_width_mm"]),
        "ref_width_px":  calib.get("ref_width_px",  DEFAULT_CALIB["ref_width_px"]),
        "model_path":    getattr(mod, "MODEL_PATH",  ""),
        "video_source":  getattr(mod, "VIDEO_SOURCE", 0),
    }


def save_config(cfg: dict):
    """Write config.py in the required format."""
    vs  = cfg["video_source"]
    vs_str = str(vs) if isinstance(vs, int) else f'r"{vs}"'
    mp  = cfg.get("model_path", "")
    mp_str = f'r"{mp}"' if mp else '""'

    lines = (
        'CALIBRATION = {\n'
        f'    "ref_height_mm": {cfg["ref_height_mm"]},\n'
        f'    "ref_height_px": {cfg["ref_height_px"]},\n'
        f'    "ref_width_mm": {cfg["ref_width_mm"]},\n'
        f'    "ref_width_px": {cfg["ref_width_px"]}\n'
        '}\n\n'
        f'MODEL_PATH = {mp_str}\n\n'
        f'VIDEO_SOURCE = {vs_str}\n'
    )
    with open(get_config_path(), "w") as f:
        f.write(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Widget helpers
# ─────────────────────────────────────────────────────────────────────────────

def flat_btn(parent, text, cmd, bg, hover, fg=None,
             font=BTN_FONT, padx=20, pady=12, **kw):
    fg = fg or C["fg"]
    b  = tk.Button(parent, text=text, command=cmd,
                   bg=bg, fg=fg, activebackground=hover, activeforeground=fg,
                   font=font, relief="flat", bd=0, cursor="hand2",
                   padx=padx, pady=pady, **kw)
    b.bind("<Enter>", lambda e: b.config(bg=hover))
    b.bind("<Leave>", lambda e: b.config(bg=bg))
    return b


def divider(parent, color=None, pady=10):
    f = tk.Frame(parent, bg=color or C["border"], height=1)
    f.pack(fill="x", pady=pady)
    return f


def field_row(parent, label, var, width=32, mono=True):
    """Label + Entry row, returns the Entry widget."""
    row = tk.Frame(parent, bg=C["card"])
    tk.Label(row, text=label, font=LABEL_FONT, fg=C["fg2"],
             bg=C["card"], width=26, anchor="w").pack(side="left")
    e = tk.Entry(row, textvariable=var,
                 font=MONO_FONT if mono else LABEL_FONT,
                 bg=C["entry"], fg=C["fg"], insertbackground=C["fg"],
                 relief="flat", bd=0, width=width,
                 highlightthickness=1, highlightbackground=C["border"],
                 highlightcolor=C["accent"])
    e.pack(side="left", ipady=6)
    return row, e


# ─────────────────────────────────────────────────────────────────────────────
# Splash Screen
# ─────────────────────────────────────────────────────────────────────────────

class Splash(tk.Toplevel):
    def __init__(self, master, on_done):
        super().__init__(master)
        self.on_done = on_done
        self.overrideredirect(True)
        self.configure(bg=C["splash"])
        W, H = 580, 320
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")
        self._build()
        self._run()

    def _build(self):
        # Outer accent border
        rim = tk.Frame(self, bg=C["accent"], padx=2, pady=2)
        rim.place(relx=0, rely=0, relwidth=1, relheight=1)
        inner = tk.Frame(rim, bg=C["splash"])
        inner.pack(fill="both", expand=True)

        # Logo image (assets/logo.png next to launcher.py)
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "ui", "assets", "logo.png")
        self._splash_logo_img = None
        try:
            from PIL import Image, ImageTk
            img = Image.open(logo_path).resize((80, 80), Image.LANCZOS)
            self._splash_logo_img = ImageTk.PhotoImage(img)
            tk.Label(inner, image=self._splash_logo_img, bg=C["splash"]).pack(pady=(28, 0))
        except Exception:
            # Fallback to emoji if PIL not available or file missing
            tk.Label(inner, text="⬡", font=("Segoe UI", 52), fg=C["accent"],
                     bg=C["splash"]).pack(pady=(28, 0))

        tk.Label(inner, text="VISION-BASED AUTOMATED FASTENER",
                 font=("Segoe UI", 16, "bold"), fg=C["fg"],
                 bg=C["splash"]).pack(pady=(6, 0))
        tk.Label(inner, text="INSPECTION SYSTEM",
                 font=("Segoe UI", 16, "bold"), fg=C["accent"],
                 bg=C["splash"]).pack()
        tk.Label(inner, text="Industrial Quality Control Platform",
                 font=SMALL_FONT, fg=C["fg3"], bg=C["splash"]).pack(pady=(3, 16))

        # Progress
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("S.Horizontal.TProgressbar",
                        troughcolor=C["border"], background=C["accent"],
                        bordercolor=C["border"], thickness=5)
        self.pb = ttk.Progressbar(inner, style="S.Horizontal.TProgressbar",
                                   mode="determinate", maximum=100, length=420)
        self.pb.pack()
        self.msg = tk.Label(inner, text="Starting…", font=SMALL_FONT,
                             fg=C["fg3"], bg=C["splash"])
        self.msg.pack(pady=(6, 0))

    def _run(self):
        steps = [
            (20,  "Loading configuration…"),
            (45,  "Initialising vision engine…"),
            (70,  "Preparing modules…"),
            (90,  "Building interface…"),
            (100, "Ready."),
        ]
        def tick(i=0):
            if i < len(steps):
                v, t = steps[i]
                self.pb["value"] = v
                self.msg.config(text=t)
                self.after(480, lambda: tick(i + 1))
            else:
                self.after(250, self._finish)
        self.after(100, tick)

    def _finish(self):
        self.destroy()
        self.on_done()


# ─────────────────────────────────────────────────────────────────────────────
# Main Application shell
# ─────────────────────────────────────────────────────────────────────────────

class App:
    WIN_W, WIN_H = 860, 580

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Vision-Based Automated Fastener Inspection")
        self.root.configure(bg=C["bg"])
        self.root.resizable(False, False)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{self.WIN_W}x{self.WIN_H}+"
                       f"{(sw-self.WIN_W)//2}+{(sh-self.WIN_H)//2}")

        self._build_header()
        self._build_content()
        self._build_statusbar()

        self.show_home()

    # ── Shell structure ────────────────────────────────────────────────

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=C["header"], height=52)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        # Logo image in header
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "ui", "assets", "logo.png")
        self._header_logo_img = None
        try:
            from PIL import Image, ImageTk
            img = Image.open(logo_path).resize((32, 32), Image.LANCZOS)
            self._header_logo_img = ImageTk.PhotoImage(img)
            tk.Label(hdr, image=self._header_logo_img, bg=C["header"]).pack(side="left", padx=(16, 6))
        except Exception:
            tk.Label(hdr, text="⬡", font=("Segoe UI", 20), fg=C["accent"],
                     bg=C["header"]).pack(side="left", padx=(16, 6))
        tk.Label(hdr, text="VISION-BASED AUTOMATED FASTENER INSPECTION",
                 font=("Segoe UI", 12, "bold"), fg=C["fg"],
                 bg=C["header"]).pack(side="left")

        # Breadcrumb label
        self.breadcrumb = tk.Label(hdr, text="", font=SMALL_FONT,
                                    fg=C["fg3"], bg=C["header"])
        self.breadcrumb.pack(side="left", padx=18)

        # Back button (hidden on home)
        self.back_btn = flat_btn(hdr, "← Back", self.show_home,
                                  bg=C["surface"], hover=C["border"],
                                  font=SMALL_FONT, padx=14, pady=4)
        self.back_btn.pack(side="right", padx=12, pady=10)

        flat_btn(hdr, "✕", self._quit, bg=C["header"],
                 hover=C["danger"], font=("Segoe UI", 11, "bold"),
                 padx=10, pady=6).pack(side="right")

        self.fix_btn = flat_btn(hdr, "🔧 Fix Paths", self._fix_paths,
                                 bg=C["surface"], hover=C["border"],
                                 font=SMALL_FONT, padx=12, pady=4)
        self.fix_btn.pack(side="right", padx=6, pady=10)

    def _build_content(self):
        """Central frame that each page renders into."""
        self.content = tk.Frame(self.root, bg=C["bg"])
        self.content.pack(fill="both", expand=True)

    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=C["header"], height=26)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        tk.Frame(bar, bg=C["accent"], width=3).pack(side="left", fill="y")
        self.status_dot  = tk.Label(bar, text="●", font=("Segoe UI", 9),
                                     fg=C["success"], bg=C["header"])
        self.status_dot.pack(side="left", padx=(8, 3))
        self.status_var  = tk.StringVar(value="Ready")
        tk.Label(bar, textvariable=self.status_var, font=SMALL_FONT,
                 fg=C["fg2"], bg=C["header"]).pack(side="left")
        tk.Label(bar, text=f"Python {sys.version.split()[0]}",
                 font=SMALL_FONT, fg=C["fg3"], bg=C["header"]).pack(side="right", padx=12)

    def _clear_content(self):
        for w in self.content.winfo_children():
            w.destroy()

    def _set_status(self, msg, color=None):
        self.status_var.set(msg)
        if color:
            self.status_dot.config(fg=color)

    def _quit(self):
        if messagebox.askyesno("Exit", "Exit the launcher?"):
            self.root.destroy()

    # ── Fix Paths ──────────────────────────────────────────────────────

    def _fix_paths(self):
        """
        Search all drives for the project folder / main.py / calibration_ui.py,
        update the global path variables, persist them into this file, and
        refresh the UI so other pages pick up the new paths immediately.
        """
        global PROJECT_FOLDER, MAIN_SCRIPT_PATH, CALIBRATION_SCRIPT_PATH

        self.fix_btn.config(state="disabled", text="🔧 Searching…")
        self._set_status("Searching drives for project files…", C["warning"])

        def worker():
            try:
                folder, main_p, cal_p, cfg_p = find_project_paths()
            except Exception as e:
                self.root.after(0, self._fix_paths_failed, str(e))
                return
            self.root.after(0, self._fix_paths_done, folder, main_p, cal_p, cfg_p)

        threading.Thread(target=worker, daemon=True).start()

    def _fix_paths_failed(self, err_msg):
        self.fix_btn.config(state="normal", text="🔧 Fix Paths")
        self._set_status("Path search failed", C["danger"])
        messagebox.showerror("Fix Paths", f"Search failed:\n{err_msg}")

    def _fix_paths_done(self, folder, main_p, cal_p, cfg_p=None):
        global PROJECT_FOLDER, MAIN_SCRIPT_PATH, CALIBRATION_SCRIPT_PATH, CONFIG_SCRIPT_PATH

        self.fix_btn.config(state="normal", text="🔧 Fix Paths")

        if not (main_p and cal_p):
            self._set_status("main.py / calibration_ui.py not found", C["danger"])
            messagebox.showerror(
                "Fix Paths",
                "Could not locate main.py and calibration_ui.py on any drive."
            )
            return

        PROJECT_FOLDER = str(folder) if folder else ""
        MAIN_SCRIPT_PATH = str(main_p)
        CALIBRATION_SCRIPT_PATH = str(cal_p)
        CONFIG_SCRIPT_PATH = str(cfg_p) if cfg_p else ""

        try:
            persist_paths_to_file(folder, main_p, cal_p, cfg_p)
        except Exception as e:
            messagebox.showwarning(
                "Fix Paths",
                f"Paths found and applied for this session, but could not be "
                f"saved to:\n{PATHS_CONFIG_FILE}\n\n{e}"
            )

        self._set_status("Paths fixed", C["success"])
        cfg_line = f"\n\nconfig.py:\n{CONFIG_SCRIPT_PATH}" if CONFIG_SCRIPT_PATH else "\n\nconfig.py: not found"
        messagebox.showinfo(
            "Fix Paths",
            f"Project folder:\n{PROJECT_FOLDER}\n\n"
            f"main.py:\n{MAIN_SCRIPT_PATH}\n\n"
            f"calibration_ui.py:\n{CALIBRATION_SCRIPT_PATH}"
            f"{cfg_line}"
        )

        # Refresh the currently open page so any displayed path updates.
        self.show_home()

    # ── Navigation ─────────────────────────────────────────────────────

    def show_home(self):
        self._clear_content()
        self.breadcrumb.config(text="")
        self.back_btn.pack_forget()
        self._build_home()
        self._set_status("Ready", C["success"])

    def _go_page(self, title, builder):
        self._clear_content()
        self.breadcrumb.config(text=f"›  {title}")
        self.back_btn.pack(side="right", padx=12, pady=10)
        builder()

    # ═══════════════════════════════════════════════════════════════════
    # HOME PAGE  –  three large nav cards
    # ═══════════════════════════════════════════════════════════════════

    def _build_home(self):
        wrap = tk.Frame(self.content, bg=C["bg"])
        wrap.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Title block
        tk.Label(wrap, text="Select a Module",
                 font=TITLE_FONT, fg=C["fg"], bg=C["bg"]).pack(pady=(48, 4))
        tk.Label(wrap, text="Choose an action to get started.",
                 font=LABEL_FONT, fg=C["fg3"], bg=C["bg"]).pack()

        divider(wrap, pady=20)

        # Card row
        card_row = tk.Frame(wrap, bg=C["bg"])
        card_row.pack(expand=True)

        cards = [
            {
                "icon":    "📐",
                "title":   "Calibration",
                "sub":     "Run cal.py to calibrate\nthe vision system",
                "color":   C["warning"],
                "hover":   C["warning_h"],
                "cmd":     lambda: self._go_page("Calibration", self._build_calibration_page),
            },
            {
                "icon":    "⚙",
                "title":   "Configuration",
                "sub":     "Set model path and\ninput source",
                "color":   C["accent"],
                "hover":   C["accent_h"],
                "cmd":     lambda: self._go_page("Configuration", self._build_config_page),
            },
            {
                "icon":    "▶",
                "title":   "Run Application",
                "sub":     "Validate settings and\nlaunch main.py",
                "color":   C["success"],
                "hover":   C["success_h"],
                "cmd":     lambda: self._launch_main(),
            },
        ]

        for c in cards:
            self._nav_card(card_row, c)

    def _nav_card(self, parent, c: dict):
        """Renders a single large navigation card."""
        W, H = 220, 200

        card = tk.Frame(parent, bg=C["card"], width=W, height=H,
                        highlightthickness=2, highlightbackground=C["border"],
                        cursor="hand2")
        card.pack(side="left", padx=16)
        card.pack_propagate(False)

        # Top accent bar
        bar = tk.Frame(card, bg=c["color"], height=5)
        bar.pack(fill="x")

        # Icon
        icon_lbl = tk.Label(card, text=c["icon"], font=("Segoe UI", 34),
                             fg=c["color"], bg=C["card"])
        icon_lbl.pack(pady=(22, 6))

        # Title
        title_lbl = tk.Label(card, text=c["title"], font=NAV_FONT,
                              fg=C["fg"], bg=C["card"])
        title_lbl.pack()

        # Subtitle
        sub_lbl = tk.Label(card, text=c["sub"], font=NAV_SUB,
                            fg=C["fg3"], bg=C["card"], justify="center")
        sub_lbl.pack(pady=(4, 16))

        # Hover / click binding on all child widgets
        def on_enter(e):
            card.config(highlightbackground=c["color"])
            bar.config(bg=c["hover"])
            icon_lbl.config(fg=c["hover"])
        def on_leave(e):
            card.config(highlightbackground=C["border"])
            bar.config(bg=c["color"])
            icon_lbl.config(fg=c["color"])
        def on_click(e):
            c["cmd"]()

        for w in (card, bar, icon_lbl, title_lbl, sub_lbl):
            w.bind("<Enter>",  on_enter)
            w.bind("<Leave>",  on_leave)
            w.bind("<Button-1>", on_click)

    # ═══════════════════════════════════════════════════════════════════
    # PAGE 1 – CALIBRATION
    # ═══════════════════════════════════════════════════════════════════

    def _build_calibration_page(self):
        self._set_status("Calibration Page", C["warning"])
        wrap = tk.Frame(self.content, bg=C["bg"])
        wrap.place(relx=0, rely=0, relwidth=1, relheight=1)

        # ── Left: info panel ─────────────────────────────────────────
        left = tk.Frame(wrap, bg=C["surface"], width=300)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        tk.Frame(left, bg=C["warning"], height=4).pack(fill="x")

        info = tk.Frame(left, bg=C["surface"])
        info.pack(fill="both", expand=True, padx=24, pady=24)

        tk.Label(info, text="📐", font=("Segoe UI", 40),
                 fg=C["warning"], bg=C["surface"]).pack(pady=(8, 4))
        tk.Label(info, text="Calibration", font=HEAD_FONT,
                 fg=C["fg"], bg=C["surface"]).pack()
        tk.Label(info, text="Launches cal.py as a\nseparate process.",
                 font=SMALL_FONT, fg=C["fg3"], bg=C["surface"],
                 justify="center").pack(pady=(6, 24))

        divider(info, color=C["border"], pady=0)

        steps = [
            ("1", "Place reference object\nin the camera view"),
            ("2", "Click Run Calibration"),
            ("3", "Follow on-screen\ninstructions in cal.py"),
            ("4", "Values auto-update\nwhen complete"),
        ]
        for num, txt in steps:
            r = tk.Frame(info, bg=C["surface"])
            r.pack(fill="x", pady=5)
            tk.Label(r, text=num, font=("Segoe UI", 9, "bold"),
                     fg=C["bg"], bg=C["warning"],
                     width=2, padx=4, pady=2).pack(side="left")
            tk.Label(r, text=txt, font=SMALL_FONT, fg=C["fg2"],
                     bg=C["surface"], justify="left").pack(side="left", padx=8)

        # ── Right: action panel ────────────────────────────────────────
        right = tk.Frame(wrap, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True)

        body = tk.Frame(right, bg=C["card"],
                        highlightthickness=1, highlightbackground=C["border"])
        body.place(relx=0.05, rely=0.08, relwidth=0.9, relheight=0.84)

        tk.Frame(body, bg=C["warning"], height=4).pack(fill="x")
        inner = tk.Frame(body, bg=C["card"])
        inner.pack(fill="both", expand=True, padx=30, pady=24)

        tk.Label(inner, text="Run Calibration Script",
                 font=HEAD_FONT, fg=C["fg"], bg=C["card"]).pack(anchor="w")
        tk.Label(inner,
                 text="This will execute  cal.py  in a separate window.\n"
                      "The launcher will wait for it to finish before updating values.",
                 font=LABEL_FONT, fg=C["fg2"], bg=C["card"],
                 justify="left").pack(anchor="w", pady=(6, 20))

        # Script path display
        cal_path = get_calibration_path()
        exists   = os.path.exists(cal_path)
        path_frame = tk.Frame(inner, bg=C["entry"],
                              highlightthickness=1,
                              highlightbackground=C["border"])
        path_frame.pack(fill="x", pady=(0, 20))
        tk.Label(path_frame,
                 text=f"  {'✔' if exists else '✘'}  {cal_path}",
                 font=MONO_FONT,
                 fg=C["success"] if exists else C["danger"],
                 bg=C["entry"], anchor="w").pack(fill="x", ipady=8, padx=6)

        if not exists:
            tk.Label(inner,
                     text="⚠  cal.py not found next to launcher.py",
                     font=SMALL_FONT, fg=C["warning"], bg=C["card"]).pack(anchor="w")

        divider(inner, pady=12)

        # Status / log area
        self._calib_status = tk.StringVar(value="Waiting to start…")
        self._calib_color  = C["fg3"]

        self.calib_log = tk.Text(inner, height=5, font=MONO_FONT,
                                  bg=C["entry"], fg=C["fg2"],
                                  relief="flat", bd=0, state="disabled",
                                  highlightthickness=1,
                                  highlightbackground=C["border"])
        self.calib_log.pack(fill="x", pady=(0, 16))
        self._log("Calibration has not been run this session.")

        # Run button
        self._calib_run_btn = flat_btn(
            inner, "⚙   Run Calibration",
            self._do_run_calibration,
            bg=C["warning"], hover=C["warning_h"],
            font=("Segoe UI", 13, "bold"),
            padx=24, pady=14
        )
        self._calib_run_btn.pack(fill="x")

    def _log(self, msg: str):
        """Append a message to the calibration log widget."""
        try:
            self.calib_log.config(state="normal")
            self.calib_log.insert("end", msg + "\n")
            self.calib_log.see("end")
            self.calib_log.config(state="disabled")
        except Exception:
            pass

    def _do_run_calibration(self):
        cal_path = get_calibration_path()
        if not os.path.exists(cal_path):
            messagebox.showerror("Not Found",
                                  f"cal.py not found:\n{cal_path}")
            return

        self._calib_run_btn.config(state="disabled", text="⏳  Running…")
        self._set_status("Running cal.py…", C["warning"])
        self._log(f"[{time.strftime('%H:%M:%S')}]  Launching cal.py …")

        def worker():
            proc = subprocess.Popen(get_python_launch_cmd(cal_path),
                                     creationflags=NO_WINDOW_FLAGS)
            proc.wait()
            self.root.after(0, self._calibration_done)

        threading.Thread(target=worker, daemon=True).start()

    def _calibration_done(self):
        self._log(f"[{time.strftime('%H:%M:%S')}]  cal.py finished.")
        self._log("Calibration complete — configuration updated.")
        self._set_status("Calibration Complete", C["success"])
        try:
            self._calib_run_btn.config(state="normal",
                                        text="⚙   Run Calibration Again")
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════════
    # PAGE 2 – CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════

    def _build_config_page(self):
        self._set_status("Configuration Page", C["accent"])

        # Load current config
        try:
            cfg = load_config()
        except Exception:
            cfg = {**DEFAULT_CALIB, "model_path": "", "video_source": 0}

        # StringVars
        self._v_model  = tk.StringVar(value=cfg.get("model_path", ""))
        self._v_source = tk.IntVar(value=0 if isinstance(cfg["video_source"], int) else 1)
        self._v_video  = tk.StringVar(
            value="" if isinstance(cfg["video_source"], int) else str(cfg["video_source"]))

        wrap = tk.Frame(self.content, bg=C["bg"])
        wrap.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Scrollable canvas
        canvas = tk.Canvas(wrap, bg=C["bg"], highlightthickness=0)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        sf = tk.Frame(canvas, bg=C["bg"])
        wid = canvas.create_window((0, 0), window=sf, anchor="nw")
        sf.bind("<Configure>", lambda e: canvas.config(
            scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(wid, width=e.width))
        canvas.bind_all("<MouseWheel>",
                         lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))

        pad = tk.Frame(sf, bg=C["bg"])
        pad.pack(fill="x", padx=40, pady=24)

        # ── Section: Model ─────────────────────────────────────────────
        self._cfg_section(pad, "⚙  Model", C["accent"])

        model_row = tk.Frame(pad, bg=C["card"])
        model_row.pack(fill="x", pady=(0, 4))
        tk.Label(model_row, text="Model Path (.pt)", font=LABEL_FONT,
                 fg=C["fg2"], bg=C["card"], width=26, anchor="w").pack(side="left")
        tk.Entry(model_row, textvariable=self._v_model, font=MONO_FONT,
                 bg=C["entry"], fg=C["fg"], insertbackground=C["fg"],
                 relief="flat", bd=0, width=36,
                 highlightthickness=1, highlightbackground=C["border"],
                 highlightcolor=C["accent"]).pack(side="left", ipady=6)
        flat_btn(model_row, "Browse", self._browse_model,
                 bg=C["surface"], hover=C["border"],
                 font=SMALL_FONT, padx=10, pady=6).pack(side="left", padx=(6, 0))
        model_row.pack(fill="x", pady=(0, 12))

        # ── Section: Video Source ──────────────────────────────────────
        self._cfg_section(pad, "🎬  Input Source", C["accent"])

        radio_row = tk.Frame(pad, bg=C["card"])
        radio_row.pack(anchor="w", pady=(0, 8))
        for lbl, val in [("📷  Camera (index 0)", 0), ("🎬  Video File", 1)]:
            tk.Radiobutton(radio_row, text=lbl, variable=self._v_source, value=val,
                           font=LABEL_FONT, fg=C["fg"], bg=C["card"],
                           selectcolor=C["entry"], activebackground=C["card"],
                           activeforeground=C["accent"],
                           command=self._toggle_video_widgets).pack(side="left", padx=(0, 24))

        vid_row = tk.Frame(pad, bg=C["card"])
        tk.Label(vid_row, text="Video File Path", font=LABEL_FONT,
                 fg=C["fg2"], bg=C["card"], width=26, anchor="w").pack(side="left")
        self._video_entry = tk.Entry(vid_row, textvariable=self._v_video, font=MONO_FONT,
                                     bg=C["entry"], fg=C["fg"], insertbackground=C["fg"],
                                     relief="flat", bd=0, width=36,
                                     highlightthickness=1, highlightbackground=C["border"],
                                     highlightcolor=C["accent"])
        self._video_entry.pack(side="left", ipady=6)
        self._video_browse = flat_btn(vid_row, "Browse", self._browse_video,
                                      bg=C["surface"], hover=C["border"],
                                      font=SMALL_FONT, padx=10, pady=6)
        self._video_browse.pack(side="left", padx=(6, 0))
        vid_row.pack(fill="x", pady=(0, 12))
        self._toggle_video_widgets()

        # ── Save button ────────────────────────────────────────────────
        divider(pad, pady=8)
        self._cfg_save_lbl = tk.Label(pad, text="", font=SMALL_FONT,
                                       fg=C["success"], bg=C["bg"])
        self._cfg_save_lbl.pack(anchor="w", pady=(0, 6))

        btn_row = tk.Frame(pad, bg=C["bg"])
        btn_row.pack(fill="x")
        flat_btn(btn_row, "💾  Save Configuration", self._save_config,
                 bg=C["accent"], hover=C["accent_h"],
                 font=("Segoe UI", 12, "bold"),
                 padx=24, pady=12).pack(side="left", padx=(0, 10))
        flat_btn(btn_row, "↺  Reset Defaults", self._reset_defaults,
                 bg=C["surface"], hover=C["border"],
                 font=BTN_FONT, padx=16, pady=12).pack(side="left")

        tk.Frame(pad, bg=C["bg"], height=30).pack()  # bottom padding

    def _cfg_section(self, parent, title, color):
        """Renders a section header."""
        f = tk.Frame(parent, bg=C["bg"])
        f.pack(fill="x", pady=(16, 6))
        tk.Label(f, text=title, font=HEAD_FONT, fg=color, bg=C["bg"]).pack(anchor="w")
        tk.Frame(f, bg=C["border"], height=1).pack(fill="x", pady=(4, 0))

    def _toggle_video_widgets(self):
        state = "normal" if self._v_source.get() == 1 else "disabled"
        try:
            self._video_entry.config(state=state)
            self._video_browse.config(state=state)
        except AttributeError:
            pass

    def _browse_model(self):
        p = filedialog.askopenfilename(title="Select Model",
                                        filetypes=[("PyTorch", "*.pt"), ("All", "*.*")])
        if p:
            self._v_model.set(p)

    def _browse_video(self):
        p = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")])
        if p:
            self._v_video.set(p)

    def _reset_defaults(self):
        if messagebox.askyesno("Reset", "Clear model path and use camera source?"):
            self._v_model.set("")
            self._v_source.set(0)
            self._v_video.set("")
            self._toggle_video_widgets()

    def _save_config(self):
        vs = self._v_video.get().strip() if self._v_source.get() == 1 else 0
        cfg = load_config()
        cfg = {
            **cfg,
            "model_path":    self._v_model.get().strip(),
            "video_source":  vs,
        }
        try:
            save_config(cfg)
            self._cfg_save_lbl.config(text="✔  Configuration saved to config.py",
                                       fg=C["success"])
            self._set_status("Configuration Saved", C["success"])
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))

    # ═══════════════════════════════════════════════════════════════════
    # PAGE 3 – RUN APPLICATION
    # ═══════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════
    # LAUNCH  –  direct launch with linked loading overlay
    # ═══════════════════════════════════════════════════════════════════

    def _launch_main(self):
        """Validate config then immediately launch main.py with a loading overlay."""
        main_path = get_main_path()

        # ── Quick silent validation ──────────────────────────────────
        if not os.path.exists(main_path):
            messagebox.showerror("Not Found", f"main.py not found:\n{main_path}")
            return
        try:
            cfg = load_config()
        except Exception as e:
            messagebox.showerror("Config Error", str(e))
            return
        mp = cfg.get("model_path", "")
        if not mp:
            messagebox.showwarning("Validation",
                "Model path is not set.\nGo to Configuration first.")
            return
        if not os.path.exists(mp):
            messagebox.showerror("Validation", f"Model file not found:\n{mp}")
            return
        vs = cfg.get("video_source", 0)
        if not isinstance(vs, int) and not os.path.exists(str(vs)):
            messagebox.showerror("Validation", f"Video file not found:\n{vs}")
            return

        # ── Show loading overlay ─────────────────────────────────────
        self._show_loading_overlay()

        # ── Start main.py and monitor startup in background thread ────
        try:
            proc = subprocess.Popen(
                get_python_launch_cmd(main_path),
                creationflags=NO_WINDOW_FLAGS,
            )
        except Exception as e:
            self._hide_loading_overlay()
            messagebox.showerror("Launch Error", str(e))
            return

        self._proc = proc
        threading.Thread(target=self._monitor_launch,
                         args=(proc,), daemon=True).start()

    def _show_loading_overlay(self):
        """Overlay a full-window loading panel on top of the current content."""
        overlay = tk.Frame(self.root, bg=C["bg"])
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._overlay = overlay

        # Card
        card = tk.Frame(overlay, bg=C["card"],
                        highlightthickness=1, highlightbackground=C["border"])
        card.place(relx=0.10, rely=0.12, relwidth=0.80, relheight=0.76)
        tk.Frame(card, bg=C["success"], height=4).pack(fill="x")

        inner = tk.Frame(card, bg=C["card"])
        inner.pack(fill="both", expand=True, padx=40, pady=30)

        # Spinning icon
        self._spin_frames = ["◐", "◓", "◑", "◒"]
        self._spin_idx = 0
        self._spin_lbl = tk.Label(inner, text="◐",
                                   font=("Segoe UI", 42), fg=C["success"],
                                   bg=C["card"])
        self._spin_lbl.pack(pady=(0, 8))

        tk.Label(inner, text="Launching Vision System",
                 font=("Segoe UI", 15, "bold"), fg=C["fg"],
                 bg=C["card"]).pack()

        self._load_msg = tk.Label(inner,
                                   text="Starting Python environment…",
                                   font=LABEL_FONT, fg=C["fg2"], bg=C["card"])
        self._load_msg.pack(pady=(6, 20))

        # Progress bar
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Launch.Horizontal.TProgressbar",
                        troughcolor=C["border"], background=C["success"],
                        bordercolor=C["border"], thickness=8)
        self._load_pb = ttk.Progressbar(inner,
                                         style="Launch.Horizontal.TProgressbar",
                                         mode="determinate",
                                         maximum=100, length=480)
        self._load_pb.pack(fill="x")

        # Step indicators
        step_frame = tk.Frame(inner, bg=C["card"])
        step_frame.pack(fill="x", pady=(18, 0))
        self._step_labels = []
        steps = ["Python", "PyTorch / YOLO", "Camera / Video", "Running"]
        for i, s in enumerate(steps):
            col = tk.Frame(step_frame, bg=C["card"])
            col.pack(side="left", expand=True)
            dot = tk.Label(col, text="○", font=("Segoe UI", 11),
                           fg=C["fg3"], bg=C["card"])
            dot.pack()
            lbl = tk.Label(col, text=s, font=SMALL_FONT,
                           fg=C["fg3"], bg=C["card"])
            lbl.pack()
            self._step_labels.append((dot, lbl))

        # Mark step 0 active immediately
        self._set_step(0, "active")
        self._load_pb["value"] = 8

        # Start spinner
        self._animate_spinner()
        self._set_status("Launching main.py…", C["success"])

    def _animate_spinner(self):
        if not hasattr(self, "_overlay") or not self._overlay.winfo_exists():
            return
        self._spin_idx = (self._spin_idx + 1) % len(self._spin_frames)
        try:
            self._spin_lbl.config(text=self._spin_frames[self._spin_idx])
            self.root.after(120, self._animate_spinner)
        except tk.TclError:
            pass

    def _set_step(self, idx, state):
        """state: active | done | pending"""
        icons  = {"active": "●", "done": "✔", "pending": "○"}
        f_clrs = {"active": C["warning"], "done": C["success"], "pending": C["fg3"]}
        t_clrs = {"active": C["fg2"],     "done": C["success"], "pending": C["fg3"]}
        dot, lbl = self._step_labels[idx]
        try:
            dot.config(text=icons[state], fg=f_clrs[state])
            lbl.config(fg=t_clrs[state])
        except tk.TclError:
            pass

    def _update_loading(self, pb_value, msg, done_step=None, active_step=None):
        try:
            self._load_pb["value"] = pb_value
            self._load_msg.config(text=msg)
            if done_step is not None:
                self._set_step(done_step, "done")
            if active_step is not None:
                self._set_step(active_step, "active")
        except (tk.TclError, AttributeError):
            pass

    def _hide_loading_overlay(self):
        try:
            self._overlay.destroy()
        except (tk.TclError, AttributeError):
            pass

    def _monitor_launch(self, proc):
        """
        Background thread: drives progress without depending on main.py output.

        The launcher cannot directly observe model/camera internals, so it
        advances through normal startup phases while checking that main.py is
        still alive.
        """
        start = time.monotonic()
        phase = -1
        phases = (
            (1.2, 22, "Importing PyTorch and YOLO...", 0, 1),
            (4.5, 55, "Loading YOLO model - opening video source...", 1, 2),
            (7.0, 82, "Video source opening - waiting for first frame...", 2, 3),
            (9.0, 100, "Vision system is now running", 3, None),
        )

        while True:
            return_code = proc.poll()
            elapsed = time.monotonic() - start

            if return_code is not None:
                if phase >= len(phases) - 1:
                    self.root.after(0, self._on_launch_complete)
                else:
                    self.root.after(0, self._on_launch_failed, return_code)
                return

            next_phase = phase + 1
            if next_phase < len(phases) and elapsed >= phases[next_phase][0]:
                phase = next_phase
                _, pb_value, msg, done_step, active_step = phases[phase]
                if phase == len(phases) - 1:
                    self.root.after(0, self._update_loading,
                                    pb_value, msg, done_step, active_step)
                    time.sleep(0.5)
                    self.root.after(0, self._on_launch_complete)
                    return
                self.root.after(0, self._update_loading,
                                pb_value, msg, done_step, active_step)

            time.sleep(0.1)

    def _on_launch_failed(self, return_code):
        """Called on main thread if main.py exits during startup."""
        self._hide_loading_overlay()
        messagebox.showerror(
            "Launch Error",
            f"main.py exited before startup completed.\nExit code: {return_code}",
        )
        self._set_status("Launch failed", C["danger"])

    def _on_launch_complete(self):
        """Called on main thread once main.py is fully running."""
        try:
            self._load_pb["value"] = 100
            self._load_msg.config(text="Vision system is now running  ✔")
            self._set_step(3, "done")
            self._spin_lbl.config(text="✔", fg=C["success"])
        except (tk.TclError, AttributeError):
            pass
        # Small pause so the user sees 100% then return to launcher menu
        def _restore():
            try:
                self._hide_loading_overlay()
            except Exception:
                pass
            try:
                self._set_status("Ready", C["success"])
            except Exception:
                pass
            try:
                self.show_home()
            except Exception:
                pass

        self.root.after(900, _restore)




# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.withdraw()

    def launch_main_window():
        root.deiconify()
        App(root)

    Splash(root, on_done=launch_main_window)
    root.mainloop()


if __name__ == "__main__":
    main()