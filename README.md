<div align="center">

# Vision Based Fastener Inspection

[![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-0078D4?style=for-the-badge&logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![ReportLab](https://img.shields.io/badge/ReportLab-PDF%20Reports-FF6B35?style=for-the-badge)](https://www.reportlab.com/)

> An industrial-grade visual inspection platform for bolt and nut measurement — combining high-performance object detection, contour-based measurement, calibration, and audit-ready PDF reporting.

</div>

---

## Table of Contents

- [Overview](#overview)
- [Business Value](#business-value)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Module Reference](#module-reference)
- [Detailed Workflow](#detailed-workflow)
- [Installation](#installation)
- [Calibration](#calibration)
- [One-Click Build & Launch](#one-click-build--launch)
- [Reporting](#reporting)
- [Recommended Deployment](#recommended-deployment)
- [Troubleshooting](#troubleshooting)

---

## Overview

![Status](https://img.shields.io/badge/Status-Production%20Ready-2ECC71?style=flat-square)
![License](https://img.shields.io/badge/License-Proprietary-E74C3C?style=flat-square)
![Inspection](https://img.shields.io/badge/Inspection-Automated-9B59B6?style=flat-square)
![Reports](https://img.shields.io/badge/Reports-ISO%20Compliant-F39C12?style=flat-square)

VBAFI is designed to:

- Detect and measure fasteners from live video input
- Convert pixel measurements to real-world millimetres
- Present a live fullscreen overlay dashboard with inspection metrics
- Generate audit-ready PDF reports with ISO lookup data
- Support a streamlined one-click Windows launch workflow

---

## Business Value

| | Advantage | Detail |
|---|---|---|
| | **Speed** | Accelerates bolt/nut inspection vs. manual gauging |
| | **Accuracy** | Reduces human error using calibrated computer vision |
| | **Auditability** | Consistent, auditable PDF reports with ISO comparisons |
| | **Deployment** | Quick setup on Windows with optional one-click launcher |
| | **Flexibility** | Configurable calibration and data sources for any line |

---

## Architecture

![Layers](https://img.shields.io/badge/Layer%201-Data%20Ingestion%20%26%20Detection-3498DB?style=flat-square)
![Layers](https://img.shields.io/badge/Layer%202-Measurement%20%26%20Validation-2ECC71?style=flat-square)
![Layers](https://img.shields.io/badge/Layer%203-Dashboard%20%26%20Reporting-E74C3C?style=flat-square)

```
  ┌─────────────────────────┐        ┌──────────────────────────┐
  │   Video / Camera Input  │        │  Calibration Reference   │
  │  (config.py VIDEO_SOURCE)│        │    (backend/config.py)   │
  └────────────┬────────────┘        └────────────┬─────────────┘
               │                                  │
               ▼                                  ▼
  ┌────────────────────────────────────────────────────────────┐
  │                     backend/main.py                        │
  │  • Loads video                                             │
  │  • Runs YOLO detection (model/yolo_bolt_nut_best.pt)       │
  │  • Crops fastener regions and derives contour masks        │
  │  • Multi-threaded: reader / processor / dashboard loop     │
  └──────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────┐   ┌──────────────────────────┐
  │  backend/texture_process_module  │◄──│   backend/calibration.py │
  │  • Mask extraction               │   │   • px → mm conversion   │
  │  • Orientation correction        │   └──────────────────────────┘
  │  • Width / height extraction     │
  └──────────────────────────┬───────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────────┐
  │              backend/industrial_dashboard.py               │
  │  • Live FULLSCREEN dashboard                               │
  │  • Overlay measurement panels                              │
  │  • PASS / FAIL status and alert button                     │
  └──────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────┐
  │    backend/report_generator.py   │
  │  • PDF creation                  │
  │  • Annotated inspection image    │
  │  • ISO lookup integration        │
  └──────────────────────────────────┘
```

---

## Repository Structure

```
Vision-Based-Fastener-Inspection
├── backend/
│   ├── launcher.py                   Tkinter GUI launcher
│   ├── main.py                       Main inspection engine
│   ├── calibration.py                px → mm converter
│   ├── calibration_ui.py             Calibration GUI
│   ├── calibration_ui_mm_decimal.py  Decimal calibration UI
│   ├── config.py                     Global configuration
│   ├── industrial_dashboard.py       Live overlay dashboard
│   ├── measurement_overlay.py        Measurement drawing utils
│   ├── optimized_VBAFI.py            Optimised pipeline variant
│   ├── report_generator.py           PDF report engine
│   └── texture_process_module.py     Image processing library
├── ISO_Metric_Database/              ISO bolt reference data
├── model/                            YOLO model weights (.pt)
├── Trail_video/                      Sample test video sources
├── ui/assets/                        UI assets for launcher
├── requirements.txt
├── initialize_app.txt                Rename to .bat to build
└── README.md
```

---

## Module Reference

### `backend/launcher.py`
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-3776AB?style=flat-square&logo=python&logoColor=white)

Provides a Tkinter startup GUI with home, calibration, configuration, and run options. Locates scripts and persists project paths in `launcher_paths.json`. Starts the main inspection pipeline without requiring command-line arguments.

---

### `backend/main.py`
![Threading](https://img.shields.io/badge/Multi--threaded-3%20threads-9B59B6?style=flat-square)
![YOLO](https://img.shields.io/badge/YOLO-Inference-00FFFF?style=flat-square&logoColor=black)

Orchestrates the full pipeline: video ingestion, model inference, measurement processing, display, and report generation.
- Multi-threaded: reader thread + processor thread + dashboard render loop
- Exponential smoothing and outlier rejection for stable real-time measurements

---

### `backend/calibration.py`
![Measurement](https://img.shields.io/badge/Measurement-px%20→%20mm-2ECC71?style=flat-square)

Converts pixel-based measurement values into real-world millimetres using a calibrated reference object. The conversion ratio is stored in `config.py`.

---

### `backend/texture_process_module.py`
![OpenCV](https://img.shields.io/badge/OpenCV-Contours-5C3EE8?style=flat-square&logo=opencv&logoColor=white)

Core measurement algorithms shared across the pipeline and calibration tools.
- Contour mask extraction from bolt detections
- Rotation correction for orientation normalisation
- Width and height extraction along measurement lines

---

### `backend/industrial_dashboard.py`
![Dashboard](https://img.shields.io/badge/Dashboard-Fullscreen-E74C3C?style=flat-square)
![Status](https://img.shields.io/badge/PASS%20%2F%20FAIL-Live%20Status-F39C12?style=flat-square)

Renders a fullscreen OpenCV dashboard overlay with orientation mask, bolt mask, measurement panels, system status, PASS/FAIL indicator, and an alert button to trigger reporting.

---

### `backend/report_generator.py`
![PDF](https://img.shields.io/badge/ReportLab-PDF-FF6B35?style=flat-square)
![ISO](https://img.shields.io/badge/ISO-Compliant-27AE60?style=flat-square)

Builds audit-ready PDF reports saved to `reports/`, including measured dimensions, ISO comparison, confidence score, and annotated inspection images.

---

### `backend/bolt_lookup.py`
![ISO](https://img.shields.io/badge/ISO%20Metric-Hex%20Bolt%20DB-1ABC9C?style=flat-square)

Matches measured dimensions to the closest bolt shape in the ISO database. Calculates a confidence score based on thread diameter, head height, and across-corners measurement.

---

### `backend/config.py`
![Config](https://img.shields.io/badge/Config-Single%20Source%20of%20Truth-95A5A6?style=flat-square)

Stores `MODEL_PATH`, `VIDEO_SOURCE`, and `CALIBRATION` values. All modules read from here — edit this file to reconfigure the system.

---

## Detailed Workflow

| Step | Badge | Action | Module |
|------|-------|--------|--------|
| 1 | ![](https://img.shields.io/badge/-Launch-3498DB?style=flat-square) | Start system via launcher or `.exe` | `launcher.py` |
| 2 | ![](https://img.shields.io/badge/-Configure-3498DB?style=flat-square) | Access calibration, config, or run mode | `launcher.py` |
| 3 | ![](https://img.shields.io/badge/-Capture-9B59B6?style=flat-square) | Open video file and begin frame capture | `main.py` |
| 4 | ![](https://img.shields.io/badge/-Detect-9B59B6?style=flat-square) | YOLO detects fasteners in sampled frames | `main.py` + model |
| 5 | ![](https://img.shields.io/badge/-Measure-2ECC71?style=flat-square) | Extract contours, correct rotation, compute lines | `texture_process_module.py` |
| 6 | ![](https://img.shields.io/badge/-Convert-2ECC71?style=flat-square) | Convert pixel values to millimetres | `calibration.py` |
| 7 | ![](https://img.shields.io/badge/-Display-F39C12?style=flat-square) | Show live annotated feed and metrics | `industrial_dashboard.py` |
| 8 | ![](https://img.shields.io/badge/-Report-E74C3C?style=flat-square) | Alert button — save PDF with ISO classification | `report_generator.py` |

---

## Installation

### System Requirements

[![Windows](https://img.shields.io/badge/Windows-10%20%2F%2011-0078D4?style=flat-square&logo=windows)](https://microsoft.com/windows)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.1%20optional-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

### Core Dependencies

[![numpy](https://img.shields.io/badge/numpy-latest-013243?style=flat-square&logo=numpy)](https://numpy.org)
[![pandas](https://img.shields.io/badge/pandas-latest-150458?style=flat-square&logo=pandas)](https://pandas.pydata.org)
[![OpenCV](https://img.shields.io/badge/opencv--python-latest-5C3EE8?style=flat-square&logo=opencv)](https://opencv.org)
[![PyTorch](https://img.shields.io/badge/torch-cu121-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Ultralytics](https://img.shields.io/badge/ultralytics-YOLO-00FFFF?style=flat-square&logoColor=black)](https://ultralytics.com)
[![ReportLab](https://img.shields.io/badge/reportlab-PDF-FF6B35?style=flat-square)](https://reportlab.com)

### Steps

**1. Create and activate a virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

**3. Confirm paths in `backend/config.py`:**
```python
MODEL_PATH   = "model/yolo_bolt_nut_best.pt"
VIDEO_SOURCE = "Trail_video/your_video.mp4"
```

**4. Verify the ISO database exists:**
```
ISO_Metric_Database/ISO_Metric_Hex_Bolt_Database.xlsx
```

---

## Calibration

> **Note:** Accurate calibration is critical for reliable mm measurements and must be completed before production use.

1. Start `backend/calibration_ui.py`
2. Select a video or image containing a **known reference object**
3. Input the real-world width and height in **millimetres**
4. Save the calibration values back into `backend/config.py`

Validate with a precision reference object before deploying in production.

---

## One-Click Build & Launch

```
1. Rename  initialize_app.txt  →  initialize_app.bat
2. Double-click  initialize_app.bat
3. Wait for build to complete
4. Double-click  Vision_Based_Fastener_Inspection.exe
```

> Right-click the `.exe` → *Send to* → *Desktop (create shortcut)* for a taskbar icon.

### Alternative Launch Options

```bash
# Run the launcher GUI
python backend/launcher.py

# Run the main inspection engine directly
python backend/main.py

# Run calibration directly
python backend/calibration_ui.py
```

---

## Reporting

![PDF](https://img.shields.io/badge/Output-PDF%20Report-FF6B35?style=flat-square)
![Folder](https://img.shields.io/badge/Saved%20to-reports%2F-27AE60?style=flat-square)

Generated PDF reports include:

- Measured bolt and thread dimensions
- ISO reference comparison and confidence score
- Annotated inspection images
- Bolt ID and summary information

---

## Recommended Deployment

- Use a **fixed camera** and **consistent lighting** for best results
- Train or validate the YOLO model for your target fastener types
- Validate calibration with a **precision reference object**
- Deploy on a **Windows workstation** with GPU acceleration if available

---

## Troubleshooting

| Symptom | Badge | Solution |
|---|---|---|
| Model load failure | ![](https://img.shields.io/badge/-Error-E74C3C?style=flat-square) | Verify `MODEL_PATH` and confirm the `.pt` file exists |
| Video not opening | ![](https://img.shields.io/badge/-Error-E74C3C?style=flat-square) | Ensure `VIDEO_SOURCE` is correct and the file is readable |
| `tkinter` errors | ![](https://img.shields.io/badge/-Warning-F39C12?style=flat-square) | Install Python with Tk/Tcl support |
| Missing packages | ![](https://img.shields.io/badge/-Warning-F39C12?style=flat-square) | Re-run `pip install -r requirements.txt` |

---

<div align="center">

![Built with](https://img.shields.io/badge/Built%20with-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Powered by](https://img.shields.io/badge/Powered%20by-YOLO%20%2B%20OpenCV-00FFFF?style=for-the-badge&logoColor=black)
![Reports](https://img.shields.io/badge/Reports-ISO%20Compliant%20PDF-FF6B35?style=for-the-badge)

</div>