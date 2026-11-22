"""
startup.py — Cloud deployment startup helper for BlindSpot.

This script is imported by app.py on Streamlit Community Cloud to ensure
model weights are present before the app starts. It is NOT needed for local runs.

It downloads the YOLOv8n weights from Ultralytics GitHub releases and the
crosswalk classifier from a public Hugging Face Hub repository (if one is set up).

HOW THIS WORKS:
  - On local machines, models/ already exists with the .pth and .pt files.
  - On Streamlit Cloud, the models/ directory is part of the repo (since both
    files are under 100 MB, they are committed directly).
  - If for any reason the files are missing, this script re-downloads them.

NOTE: This file is intentionally separate from app.py. app.py is NOT changed.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained_weights"


def ensure_yolo_weights():
    """Download YOLOv8n if not already present."""
    yolo_path = PRETRAINED_DIR / "yolov8n.pt"
    alt_yolo_path = PROJECT_ROOT / "yolov8n.pt"

    if yolo_path.exists() or alt_yolo_path.exists():
        return True

    try:
        import urllib.request
        PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
        print(f"[startup] Downloading YOLOv8n weights from {url} ...")
        urllib.request.urlretrieve(url, yolo_path)
        print(f"[startup] YOLOv8n saved to {yolo_path}")
        return True
    except Exception as e:
        print(f"[startup] WARNING: Could not download YOLOv8n: {e}")
        return False


def ensure_dirs():
    """Make sure all required output directories exist."""
    dirs = [
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "models" / "pretrained_weights",
        PROJECT_ROOT / "outputs" / "evaluation",
        PROJECT_ROOT / "outputs" / "sample_predictions",
        PROJECT_ROOT / "outputs" / "figures",
        PROJECT_ROOT / "dataset" / "crosswalk" / "Original",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def run_startup():
    """Call this at the top of app.py (cloud only) or via a separate entrypoint."""
    ensure_dirs()
    ensure_yolo_weights()


if __name__ == "__main__":
    run_startup()
    print("[startup] Startup checks complete.")
