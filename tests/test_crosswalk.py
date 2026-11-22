"""
Unit and Integration Tests for BlindSpot Crosswalk Recognition
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from modules.crosswalk_recognition import CrosswalkRecognizer
from modules.image_processing import run_image_processing_pipeline, analyze_stripe_patterns
import config

@pytest.fixture(scope="module")
def recognizer():
    return CrosswalkRecognizer(model_path=str(config.CROSSWALK_MODEL_PATH))

def test_recognizer_initialization(recognizer):
    assert recognizer is not None
    assert recognizer.model is not None

def test_dip_pipeline_stages():
    dummy = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
    pipeline = run_image_processing_pipeline(dummy)
    expected_keys = [
        'original', 'grayscale', 'blurred', 'otsu_thresh',
        'adaptive_thresh', 'canny_edges', 'canny_roi', 'morph_clean', 'morph_roi'
    ]
    for k in expected_keys:
        assert k in pipeline
        assert isinstance(pipeline[k], np.ndarray)

def test_stripe_pattern_analysis():
    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
    # Paint 4 parallel rectangular white stripes to simulate zebra stripes in lower half
    # Each stripe: width 120, height 25 (aspect ratio 4.8), centered horizontally
    for y in [180, 220, 260, 300]:
        cv2.rectangle(dummy, (240, y), (400, y + 25), (255, 255, 255), -1)
        
    analysis = analyze_stripe_patterns(dummy)
    assert 'candidate_stripes_count' in analysis
    assert 'stripe_confidence' in analysis
    assert analysis['candidate_stripes_count'] > 0

def test_predict_positive_crosswalk(recognizer):
    pos_path = Path("dataset/crosswalk/Original/c1_p1_1.jpg")
    if pos_path.exists():
        img = cv2.imread(str(pos_path))
        res = recognizer.predict(img)
        assert res['predicted_class_idx'] in [0, 1, 2]
        assert res['crossing_detected'] is True
        assert "Possible zebra crossing" in res['status_text']
        assert res['latency_ms'] > 0

def test_predict_negative_crosswalk(recognizer):
    neg_path = Path("dataset/crosswalk/Original/c4_p1_2.jpg")
    if neg_path.exists():
        img = cv2.imread(str(neg_path))
        res = recognizer.predict(img)
        assert res['predicted_class_idx'] == 3
        assert res['crossing_detected'] is False
        assert "No clear zebra crossing" in res['status_text']
