"""
Multimodal Decision Integration Layer for BlindSpot
Fuses visual perception outputs from:
1. Spatial Obstacle Detection (YOLOv8n)
2. Zebra Crossing Recognition (MobileNetV3 + OpenCV DIP)

Produces:
- Unified decision-level alert
- Multi-channel prioritized spoken feedback
- Comprehensive composite visual rendering (bounding boxes, spatial zones, crossing badge, walkable corridor)

Note on Architecture:
This is a lightweight decision-level fusion layer that integrates independent visual perception
modalities into coherent assistive cues, avoiding cognitive audio overload for the user.
"""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np
import time

import config
from modules.obstacle_detection import ObstacleDetector
from modules.crosswalk_recognition import CrosswalkRecognizer
from modules.voice_assistance import generate_speech_audio

class BlindSpotSystem:
    def __init__(self,
                 yolo_weights: Optional[str] = None,
                 crosswalk_weights: Optional[str] = None,
                 conf_threshold: float = config.DEFAULT_CONF_THRESHOLD):
        print("Initializing integrated BlindSpot perception system...")
        self.obstacle_detector = ObstacleDetector(weights_path=yolo_weights, conf_threshold=conf_threshold)
        self.crosswalk_recognizer = CrosswalkRecognizer(model_path=crosswalk_weights)

    def process_frame(self,
                      image_bgr: np.ndarray,
                      conf_threshold: Optional[float] = None,
                      include_walking_corridor: bool = True) -> Dict:
        """
        Processes a single visual scene through both perception pipelines and performs decision fusion.
        """
        t0 = time.perf_counter()
        
        # 1. Obstacle Detection Pipeline
        obs_res = self.obstacle_detector.detect(image_bgr, conf_threshold=conf_threshold)
        
        # 2. Crosswalk Recognition Pipeline
        cw_res = self.crosswalk_recognizer.predict(image_bgr, include_path_awareness=include_walking_corridor)
        
        # 3. Decision-Level Fusion & Spoken Alert Formulation
        fused_alert, priority_level = self.fuse_alerts(obs_res, cw_res)
        
        # 4. Generate composite annotated visualization
        composite_image = self.render_composite_overlay(
            image_bgr, obs_res, cw_res, fused_alert, include_corridor=include_walking_corridor
        )
        
        t1 = time.perf_counter()
        total_latency_ms = (t1 - t0) * 1000.0
        
        # 5. Synthesize voice audio
        audio_bytes = generate_speech_audio(fused_alert)
        
        return {
            'obstacles': obs_res['detected_objects'],
            'obstacle_count': obs_res['count'],
            'crosswalk': {
                'predicted_class': cw_res['predicted_class_name'],
                'confidence': cw_res['confidence'],
                'crossing_detected': cw_res['crossing_detected'],
                'view_direction': cw_res['view_direction'],
                'status_text': cw_res['status_text']
            },
            'fused_spoken_alert': fused_alert,
            'alert_priority': priority_level,
            'audio_bytes': audio_bytes,
            'composite_image': composite_image,
            'obstacle_annotated': obs_res['annotated_image'],
            'crosswalk_annotated': cw_res['annotated_image'],
            'dip_stages': cw_res['pipeline_stages'],
            'latencies': {
                'yolo_ms': obs_res['latency_ms'],
                'crosswalk_ms': cw_res['latency_ms'],
                'total_ms': round(total_latency_ms, 2)
            },
            'fps': round(1000.0 / (total_latency_ms + 1e-5), 1)
        }

    def fuse_alerts(self, obs_res: Dict, cw_res: Dict) -> Tuple[str, str]:
        """
        Decision-level arbitration logic:
        Priority 1 (HIGH): Immediate obstacle in centre walking path.
        Priority 2 (MEDIUM): Flank obstacles (left/right) OR crossing detected.
        Priority 3 (LOW): No obstacles detected.
        
        CRITICAL RULE: Never claim the path is clear if obstacles are detected!
        """
        detected_objects = obs_res.get('detected_objects', [])
        centre_obs = [o for o in detected_objects if o['zone'] == 'centre']
        left_obs = [o for o in detected_objects if o['zone'] == 'left']
        right_obs = [o for o in detected_objects if o['zone'] == 'right']
        
        crossing_detected = cw_res.get('crossing_detected', False)
        cw_status = cw_res.get('status_text', 'No clear zebra crossing detected.')
        
        components = []
        priority = "LOW"
        
        # Case A: Obstacles detected in the scene
        if detected_objects:
            # 1. Centre Obstacle Hazard (Collision Warning)
            if centre_obs:
                priority = "HIGH"
                first_name = centre_obs[0]['class_name']
                desc = "vehicle" if first_name in ['car', 'bus', 'truck'] else first_name
                if len(centre_obs) > 1:
                    components.append(f"Caution: Multiple obstacles detected in your centre walking path.")
                else:
                    components.append(f"Caution: {desc.capitalize()} detected in your centre path.")
            
            # 2. Side Obstacles (Flank Spatial Awareness)
            if left_obs and len(components) < 2:
                if priority != "HIGH":
                    priority = "MEDIUM"
                desc = "vehicle" if left_obs[0]['class_name'] in ['car', 'bus', 'truck'] else left_obs[0]['class_name']
                components.append(f"{desc.capitalize()} on your left.")
                
            if right_obs and len(components) < 2:
                if priority != "HIGH":
                    priority = "MEDIUM"
                desc = "vehicle" if right_obs[0]['class_name'] in ['car', 'bus', 'truck'] else right_obs[0]['class_name']
                components.append(f"{desc.capitalize()} on your right.")
                
            # Append crosswalk status if crossing is visible
            if crossing_detected:
                components.append(cw_status)
                
            return " ".join(components), priority

        # Case B: Zero obstacles detected anywhere in the frame
        else:
            if crossing_detected:
                return f"No obstacles detected. {cw_status}", "MEDIUM"
            else:
                return f"No obstacles detected in view. {cw_status}", "LOW"

    def render_composite_overlay(self,
                                  image_bgr: np.ndarray,
                                  obs_res: Dict,
                                  cw_res: Dict,
                                  alert_text: str,
                                  include_corridor: bool = True) -> np.ndarray:
        """Combines YOLO bounding boxes, spatial lines, and crosswalk HUD on the original image."""
        composite = image_bgr.copy()
        h, w = composite.shape[:2]
        
        # 1. Spatial boundary guidelines
        x_left = int(w * config.LEFT_ZONE_BOUNDARY)
        x_right = int(w * config.RIGHT_ZONE_BOUNDARY)
        for y in range(0, h, 20):
            cv2.line(composite, (x_left, y), (x_left, y + 10), (180, 180, 180), 1)
            cv2.line(composite, (x_right, y), (x_right, y + 10), (180, 180, 180), 1)
            
        # Zone tags at top
        cv2.putText(composite, "LEFT ZONE", (x_left // 2 - 45, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 255), 1)
        cv2.putText(composite, "CENTRE PATH", ((x_left + x_right) // 2 - 50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 255, 150), 1)
        cv2.putText(composite, "RIGHT ZONE", ((x_right + w) // 2 - 45, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 255), 1)

        # 2. Walking corridor if enabled
        if include_corridor:
            from modules.image_processing import generate_walkable_path_awareness
            composite = generate_walkable_path_awareness(composite)

        # 3. Obstacle bounding boxes
        for obj in obs_res.get('detected_objects', []):
            x1, y1, x2, y2 = obj['bbox']
            zone = obj['zone']
            conf = obj['confidence']
            name = obj['class_name']
            color = (0, 140, 255) if zone == 'centre' else (255, 200, 0)
            cv2.rectangle(composite, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf*100:.0f}% ({zone})"
            cv2.putText(composite, label, (x1 + 3, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # 4. Composite Header Alert Banner
        cv2.rectangle(composite, (0, 0), (w, 40), (25, 25, 25), -1)
        cv2.putText(composite, f"BlindSpot Alert: {alert_text[:95]}", (15, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 255), 1, cv2.LINE_AA)
        
        # 5. Bottom Crosswalk Status Bar
        cw_class = cw_res.get('predicted_class_name', 'Unknown')
        cw_conf = cw_res.get('confidence', 0.0)
        cv2.rectangle(composite, (0, h - 35), (w, h), (20, 20, 20), -1)
        cv2.putText(composite, f"Crosswalk: {cw_class} ({cw_conf*100:.1f}%) | Latency: {obs_res.get('latency_ms', 0) + cw_res.get('latency_ms', 0):.1f}ms",
                    (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        return composite
