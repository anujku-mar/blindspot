"""
Obstacle Detection Module for BlindSpot
Powered by Pretrained YOLOv8-nano on COCO.
Performs:
- Object detection for pedestrian navigation
- Bounding box and confidence rendering
- 3-zone horizontal spatial partitioning (Left, Centre, Right)
- Spatial alert formulation with deduplication and cooldown
- Honest disclaimer: no distance estimation or collision prediction claimed
"""

import time
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

import config

class ObstacleDetector:
    def __init__(self, weights_path: Optional[str] = None, conf_threshold: float = config.DEFAULT_CONF_THRESHOLD):
        self.weights_path = weights_path or str(config.YOLO_WEIGHTS_PATH)
        self.conf_threshold = conf_threshold
        self.iou_threshold = config.DEFAULT_IOU_THRESHOLD
        
        # Cooldown management for video streaming
        self.last_alert_time = 0.0
        self.last_alert_message = ""
        self.cooldown_seconds = config.ALERT_COOLDOWN_SECONDS
        
        # Load model
        print(f"Loading YOLO model from: {self.weights_path}")
        self.model = YOLO(self.weights_path)

    def determine_zone(self, cx: float, width: int) -> str:
        """Assign object horizontal zone: 'left', 'centre', or 'right'."""
        rel_x = cx / float(width)
        if rel_x < config.LEFT_ZONE_BOUNDARY:
            return "left"
        elif rel_x <= config.RIGHT_ZONE_BOUNDARY:
            return "centre"
        else:
            return "right"

    def detect(self, image_bgr: np.ndarray, conf_threshold: Optional[float] = None) -> Dict:
        """
        Run inference on image_bgr.
        Detects any physical obstacles in the field of view without artificially dropping classes.
        Returns detection results, annotated image, and spatial voice alerts.
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        h, w = image_bgr.shape[:2]
        
        t0 = time.perf_counter()
        # Predict on all COCO classes so no real-world obstacle is ignored
        results = self.model.predict(
            source=image_bgr,
            conf=conf,
            iou=self.iou_threshold,
            verbose=False
        )
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        annotated_image = image_bgr.copy()
        detected_objects = []
        
        # Draw spatial dividing guidelines (Left, Centre, Right zones)
        x_left = int(w * config.LEFT_ZONE_BOUNDARY)
        x_right = int(w * config.RIGHT_ZONE_BOUNDARY)
        
        # Dotted vertical boundary lines
        for y in range(0, h, 20):
            cv2.line(annotated_image, (x_left, y), (x_left, y + 10), (180, 180, 180), 1)
            cv2.line(annotated_image, (x_right, y), (x_right, y + 10), (180, 180, 180), 1)
            
        # Zone headers at top
        cv2.putText(annotated_image, "LEFT ZONE", (x_left // 2 - 45, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 255), 2)
        cv2.putText(annotated_image, "CENTRE PATH", ((x_left + x_right) // 2 - 50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 2)
        cv2.putText(annotated_image, "RIGHT ZONE", ((x_right + w) // 2 - 45, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 255), 2)

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names.get(cls_id, f"object_{cls_id}")
                confidence = float(box.conf[0].cpu().numpy())
                
                # Bounding box center
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                zone = self.determine_zone(cx, w)
                
                detected_objects.append({
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'confidence': round(confidence, 3),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'center': (round(float(cx), 1), round(float(cy), 1)),
                    'zone': zone
                })
                
                # Zone-specific bounding box colors (Centre: bright orange/red, Sides: cyan/yellow)
                color = (0, 140, 255) if zone == "centre" else (255, 200, 0)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                label_text = f"{cls_name.capitalize()} {confidence*100:.0f}% [{zone.upper()}]"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_image, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(th + 6, y1)), color, -1)
                cv2.putText(annotated_image, label_text, (x1 + 2, max(th, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Formulate concise spatial voice warnings
        voice_alert = self.generate_spatial_alert(detected_objects)
        
        return {
            'detected_objects': detected_objects,
            'count': len(detected_objects),
            'voice_alert': voice_alert,
            'annotated_image': annotated_image,
            'latency_ms': round(latency_ms, 2),
            'fps': round(1000.0 / (latency_ms + 1e-5), 1)
        }

    def generate_spatial_alert(self, detected_objects: List[Dict]) -> str:
        """
        Formulate clear, concise spoken feedback.
        Rules:
        - Alert centre obstacles first (pedestrian path).
        - Explicitly state side obstacles (left/right).
        - Never say path is clear if obstacles exist!
        """
        if not detected_objects:
            return "No obstacles detected in your immediate view."
            
        # Group objects by zone
        zone_groups = {'centre': [], 'left': [], 'right': []}
        for obj in detected_objects:
            zone_groups[obj['zone']].append(obj['class_name'])
            
        messages = []
        
        # High priority: centre zone (direct walking collision hazard)
        if zone_groups['centre']:
            first_obj = zone_groups['centre'][0]
            obj_desc = "vehicle" if first_obj in ['car', 'bus', 'truck'] else first_obj
            if len(zone_groups['centre']) > 1:
                messages.append(f"Caution: Multiple obstacles detected in your centre walking path.")
            else:
                messages.append(f"Caution: {obj_desc.capitalize()} detected in your centre walking path.")
                
        # Medium priority: left / right flank obstacles
        if zone_groups['left']:
            first_obj = zone_groups['left'][0]
            obj_desc = "vehicle" if first_obj in ['car', 'bus', 'truck'] else first_obj
            messages.append(f"{obj_desc.capitalize()} detected on your left.")
            
        if zone_groups['right']:
            first_obj = zone_groups['right'][0]
            obj_desc = "vehicle" if first_obj in ['car', 'bus', 'truck'] else first_obj
            messages.append(f"{obj_desc.capitalize()} detected on your right.")
            
        # If very crowded
        if len(detected_objects) >= 4:
            return f"Caution: Crowded scene ahead with {len(detected_objects)} obstacles detected around your walking path."
            
        return " ".join(messages[:config.MAX_CONCURRENT_ALERTS])

    def should_speak(self, current_alert: str) -> bool:
        """Cooldown and deduplication check for video streaming."""
        now = time.time()
        if (now - self.last_alert_time > self.cooldown_seconds) or (current_alert != self.last_alert_message):
            self.last_alert_time = now
            self.last_alert_message = current_alert
            return True
        return False
