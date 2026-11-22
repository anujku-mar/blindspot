"""
BlindSpot System Configuration
Central configuration for model weights, thresholds, paths, class mappings, and spatial zones.
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR'] = str((BASE_DIR / ".ultralytics").resolve())
os.environ['MPLCONFIGDIR'] = str((BASE_DIR / ".matplotlib").resolve())
(BASE_DIR / ".ultralytics").mkdir(exist_ok=True)
(BASE_DIR / ".matplotlib").mkdir(exist_ok=True)

MODELS_DIR = BASE_DIR / "models"
PRETRAINED_WEIGHTS_DIR = MODELS_DIR / "pretrained_weights"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATASET_DIR = BASE_DIR / "dataset"

YOLO_WEIGHTS_PATH = PRETRAINED_WEIGHTS_DIR / "yolov8n.pt"
CROSSWALK_MODEL_PATH = MODELS_DIR / "crosswalk_classifier.pth"
CROSSWALK_MANIFEST_PATH = DATASET_DIR / "dataset_manifest.json"
EVALUATION_METRICS_PATH = OUTPUTS_DIR / "evaluation" / "crosswalk_metrics.json"

# Object Detection Configuration
# Pedestrian-relevant street obstacles (people, vehicles, animals, street furniture, barriers, common items)
# Any object detected by YOLO is treated as a physical obstacle for a blind pedestrian
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    6: "train",
    7: "truck",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
    56: "chair",
    57: "couch",
    58: "potted plant",
    62: "tv",
    63: "laptop",
    73: "book"
}

# Default confidence threshold - 0.22 ensures small/distant street obstacles are detected reliably
DEFAULT_CONF_THRESHOLD = 0.22
DEFAULT_IOU_THRESHOLD = 0.45

# Spatial Zoning Configuration (horizontal division)
# Left zone: [0.0, LEFT_ZONE_BOUNDARY)
# Centre zone: [LEFT_ZONE_BOUNDARY, RIGHT_ZONE_BOUNDARY]
# Right zone: (RIGHT_ZONE_BOUNDARY, 1.0]
LEFT_ZONE_BOUNDARY = 0.33
RIGHT_ZONE_BOUNDARY = 0.67

# Crosswalk Recognition Configuration
CROSSWALK_CLASSES = {
    0: "Front-View Crossing",
    1: "Left-Side Perspective",
    2: "Right-Side Perspective",
    3: "Non-Crosswalk"
}

CROSSWALK_CONF_THRESHOLD = 0.50

# Voice Alert Configuration
ALERT_COOLDOWN_SECONDS = 3.0  # deduplication cooldown for video streams
MAX_CONCURRENT_ALERTS = 3     # prevent speech overload
