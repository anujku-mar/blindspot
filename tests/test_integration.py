"""
System Integration Tests for BlindSpot Multimodal Pipeline
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from modules.integration import BlindSpotSystem
from modules.voice_assistance import generate_speech_audio

@pytest.fixture(scope="module")
def system():
    return BlindSpotSystem()

def test_system_initialization(system):
    assert system.obstacle_detector is not None
    assert system.crosswalk_recognizer is not None

def test_process_frame_end_to_end(system):
    # Test on a real crosswalk image
    sample_path = Path("dataset/crosswalk/Original/c1_p1_10.jpg")
    if sample_path.exists():
        img = cv2.imread(str(sample_path))
        result = system.process_frame(img)
        
        # Verify structure
        assert 'obstacles' in result
        assert 'crosswalk' in result
        assert 'fused_spoken_alert' in result
        assert 'alert_priority' in result
        assert 'composite_image' in result
        assert 'latencies' in result
        assert result['composite_image'].shape == img.shape
        assert result['alert_priority'] in ['HIGH', 'MEDIUM', 'LOW']

def test_fuse_alerts_priority(system):
    # Mock high-priority scenario: car in centre
    mock_obs = {
        'detected_objects': [{'class_name': 'car', 'zone': 'centre', 'confidence': 0.9}]
    }
    mock_cw = {
        'crossing_detected': True,
        'status_text': 'Possible zebra crossing detected ahead.'
    }
    alert, priority = system.fuse_alerts(mock_obs, mock_cw)
    assert priority == "HIGH"
    assert "Caution: Vehicle detected in your centre path" in alert

def test_audio_generation():
    text = "Caution: Person detected on your left."
    # Generate speech
    audio = generate_speech_audio(text)
    # Even if offline/sandbox has no network, should return bytes or None without crashing
    if audio is not None:
        assert isinstance(audio, bytes)
        assert len(audio) > 0
