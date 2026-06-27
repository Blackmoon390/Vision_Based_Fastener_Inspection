Vision Based Fastener Inspection
================================

Overview
--------
Vision Based Fastener Inspection is an industrial-grade visual inspection platform for bolt and nut measurement. It combines high-performance object detection, contour-based measurement, calibration, and report generation to support automated quality assurance in manufacturing and assembly environments.

The system is designed to:
- detect and measure fasteners from video input
- convert pixel measurements to real-world millimeters
- present a live overlay dashboard with inspection metrics
- generate audit-ready PDF reports with ISO lookup data
- support a streamlined Windows launch workflow

Architecture
------------
The system is composed of three main layers:

  1. Data ingestion and detection
  2. Measurement and validation
  3. Dashboard display and reporting

ASCII Architecture Diagram
--------------------------

  +---------------------------+        +------------------------+
  | Video / Camera Input      |        | Calibration Reference  |
  | (backend/config.py VIDEO_SOURCE) |        | (backend/config.py)    |
  +-----------+---------------+        +-----------+------------+
              |                                    |
              v                                    v
  +-------------------------------------------------------+
  | backend/main.py                                        |
  | - loads video                                          |
  | - runs YOLO detection with backend/model/yolo...pt     |
  | - crops fastener regions and derives contour masks     |
  +-------------------------------------------------------+
              |
              v
  +-----------------------------+     +----------------------------+
  | backend/texture_process_module.py |<--| backend/calibration.py    |
  | - mask extraction                 |   | - px-to-mm conversion     |
  | - orientation correction          |   +----------------------------+
  | - width/height extraction         |
  +-----------------------------+
              |
              v
  +-------------------------------------------------------+
  | backend/industrial_dashboard.py                     |
  | - live FULLSCREEN dashboard                          |
  | - overlay measurement panels                         |
  | - PASS/FAIL status and alert button                  |
  +-------------------------------------------------------+
              |
              v
  +-----------------------------+
  | backend/report_generator.py  |
  | - PDF creation               |
  | - annotated inspection image  |
  | - ISO lookup integration      |
  +-----------------------------+

Repository Structure
--------------------
backend/                Core application modules and GUI tools
    bolt_lookup.py      ISO database lookup for measured bolt sizes
    calibration.py      Calibration converter for px → mm measurements
    calibration_ui.py   Calibration GUI for pixel reference setup
    calibration_ui_mm_decimal.py  Decimal-friendly calibration interface
    config.py           Global configuration values
    industrial_dashboard.py  Render live overlay dashboard
    launcher.py         Tkinter launcher for startup and configuration
    main.py             Main inspection engine
    measurement_overlay.py  Annotated measurement drawing utilities
    optimized_VBAFI.py  Optimized pipeline variant
    report_generator.py PDF report engine
    texture_process_module.py  Image processing and measurement helper library
ISO_Metric_Database/    ISO bolt reference database used by bolt_lookup
model/                  YOLO model weights for detection
Trail_video/            Sample or test video sources
ui/assets/              UI assets for the launcher and installer
requirements.txt        Python dependency list and install notes
initialize_app.txt      Startup/build helper text file
initialize_app.bat      Optional batch launcher for easy startup
README.txt             This document

Business Value
--------------
- Accelerates quality inspection for fastener assembly.
- Reduces manual measurement errors with calibrated vision-based metrics.
- Produces formal reports suitable for audit and traceability.
- Supports rapid deployment on Windows workstations.

Requirements
------------
- Windows 10 or Windows 11
- Python 3.9 or newer
- Optional NVIDIA GPU with CUDA 12.1 for accelerated inference
- Dependencies installed from `requirements.txt`

Core dependencies:
- numpy
- pandas
- opencv-python
- Pillow
- torch
- ultralytics
- reportlab
- pyinstaller

Installation
------------
1. Create and activate a Python virtual environment:
   python -m venv venv
   venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

3. Confirm the following paths in `backend/config.py`:
   - `MODEL_PATH` points to `model/yolo_bolt_nut_best.pt`
   - `VIDEO_SOURCE` points to a valid `.mp4` file in `Trail_video/`

4. Verify `ISO_Metric_Database/ISO_Metric_Hex_Bolt_Database.xlsx` exists for ISO lookups.

Startup Workflow
----------------
For a one-click Windows startup flow:
1. Rename `initialize_app.txt` to `initialize_app.bat`.
2. Double-click `initialize_app.bat`.
3. The batch file builds the application executable and places `Vision_Based_Fastener_Inspection.exe` in the project root.
4. After the build completes, double-click `Vision_Based_Fastener_Inspection.exe` or create a Windows shortcut to run the app.

This makes the system easy to launch as an application icon from Windows Explorer.

Alternative launch options:
- Run the launcher directly:
  python backend/launcher.py
- Run the main inspection engine directly:
  python backend/main.py
- Run calibration directly:
  python backend/calibration_ui.py

Calibration
-----------
Before production use, calibrate the system:
1. Start `backend/calibration_ui.py`.
2. Select a video or image containing a known reference object.
3. Input the real-world width and height in millimeters.
4. Save the calibration values back into `backend/config.py`.

This ensures measurements convert accurately from pixels to millimeters.

How it works
-------------
1. `backend/main.py` opens the configured video source.
2. YOLO detects fasteners in each frame.
3. `texture_process_module.py` extracts the largest contour, corrects orientation, and computes widths/heights.
4. `backend/calibration.py` converts pixel measurements into physical millimeters.
5. The overlay dashboard shows live metrics and status.
6. The alert button triggers `backend/report_generator.py` to create a PDF report.

Reporting
---------
Generated PDF reports are saved to the `reports/` folder.
Reports include:
- Measured bolt and thread dimensions
- ISO reference comparison and confidence
- Annotated inspection images
- Bolt ID and summary information

Recommended Deployment
----------------------
- Use a fixed camera and consistent lighting for best results.
- Train or validate the YOLO model for the target fastener types.
- Validate calibration with a precision reference object.
- Deploy on a Windows workstation with GPU acceleration if available.

Troubleshooting
---------------
- `Model load failure`: verify `MODEL_PATH` and that the `.pt` file exists.
- `Video not opening`: ensure `VIDEO_SOURCE` is correct and readable.
- `tkinter errors`: install Python with Tk/Tcl support or use a distribution that includes Tk.
- `Missing packages`: reinstall with `pip install -r requirements.txt`.

Appendix
--------
- `backend/launcher.py` provides a GUI entry point for calibration, configuration, and running the inspection.
- `backend/industrial_dashboard.py` renders a fullscreen overlay dashboard from live video frames.
- `backend/report_generator.py` produces PDF reports designed for quality records.
