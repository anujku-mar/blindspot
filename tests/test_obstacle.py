"""
Unit and Integration Tests for BlindSpot Obstacle Detection
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from modules.obstacle_detection import ObstacleDetector
import config

@pytest.fixture(scope="module")
def detector():
    return ObstacleDetector(weights_path=str(config.YOLO_WEIGHTS_PATH), conf_threshold=0.25)

def test_detector_initialization(detector):
    assert detector is not None
    assert detector.model is not None
    assert detector.conf_threshold == 0.25

def test_determine_zone(detector):
    width = 1000
    # Left zone (< 330)
    assert detector.determine_zone(150, width) == "left"
    # Centre zone (330 to 670)
    assert detector.determine_zone(500, width) == "centre"
    # Right zone (> 670)
    assert detector.determine_zone(850, width) == "right"

def test_detect_blank_image(detector):
    # Blank black image should return 0 detections
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect(blank)
    assert res['count'] == 0
    assert ("no obstacles" in res['voice_alert'].lower()) or ("clear" in res['voice_alert'].lower())
    assert res['annotated_image'].shape == (480, 640, 3)

def test_detect_sample_dataset_image(detector):
    # Test on a real street scene from crosswalk dataset
    sample_path = Path("dataset/crosswalk/Original/c1_p1_1.jpg")
    if sample_path.exists():
        img = cv2.imread(str(sample_path))
        res = detector.detect(img)
        assert 'detected_objects' in res
        assert 'annotated_image' in res
        assert 'latency_ms' in res
        assert res['latency_ms'] > 0

def test_alert_generation(detector):
    dummy_objects = [
        {'class_name': 'person', 'zone': 'left', 'confidence': 0.85},
        {'class_name': 'car', 'zone': 'centre', 'confidence': 0.90}
    ]
    alert = detector.generate_spatial_alert(dummy_objects)
    assert "centre" in alert.lower()
    assert "vehicle" in alert.lower()
    assert "left" in alert.lower()
    assert "person" in alert.lower()

def test_cooldown_mechanism(detector):
    alert = "Person detected on your left."
    # First time should speak
    should1 = detector.should_speak(alert)
    assert should1 is True
    # Immediate repeat of same alert should be suppressed
    should2 = detector.should_speak(alert)
    assert should2 is False
