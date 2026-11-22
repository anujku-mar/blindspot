"""
Crosswalk Recognition Module for BlindSpot
Combines:
1. Fine-tuned Deep Learning Classifier (MobileNetV3-Small on Kaggle Crosswalk Dataset)
2. Classical OpenCV Stripe Pattern Analysis (DIP filters, morphological opening, contour periodicity)
3. Walking-path visual awareness
4. Safe, non-assertive spoken feedback
"""

import time
from typing import Dict, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models

import config
from modules.image_processing import analyze_stripe_patterns, generate_walkable_path_awareness

class CrosswalkRecognizer:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or str(config.CROSSWALK_MODEL_PATH)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = config.CROSSWALK_CLASSES
        self.conf_threshold = config.CROSSWALK_CONF_THRESHOLD
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load MobileNetV3-Small architecture
        try:
            self.model = models.mobilenet_v3_small(weights=None)
        except Exception:
            self.model = models.mobilenet_v3_small()
            
        num_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 4)
        )
        
        if Path(self.model_path).exists():
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"Loaded trained crosswalk model from: {self.model_path}")
        else:
            print(f"Warning: Model weights not found at {self.model_path}. Fallback to random weights.")
            
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, image_bgr: np.ndarray, include_path_awareness: bool = True) -> Dict:
        """
        Runs hybrid prediction:
        - Deep learning inference
        - Classical stripe periodicity verification
        - Walking corridor overlay
        """
        h, w = image_bgr.shape[:2]
        
        # 1. Classical OpenCV Stripe Analysis
        classical_res = analyze_stripe_patterns(image_bgr)
        stripe_conf = classical_res['stripe_confidence']
        stripes_count = classical_res['candidate_stripes_count']
        
        # 2. Deep Learning Classifier Inference
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(tensor_img)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_class_idx = int(np.argmax(probs))
            pred_conf = float(probs[pred_class_idx])
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        
        pred_class_name = self.class_names[pred_class_idx]
        
        # 3. Hybrid Decision & Spoken Feedback formulation
        # Safety rule: If confidence is marginal, or conflicting signals exist, state uncertainty
        if pred_conf < 0.50:
            status_text = "Crossing detection uncertain. Please verify your surroundings."
            crossing_detected = False
            view_direction = "uncertain"
        elif pred_class_idx == 3:  # Non-crosswalk
            status_text = "No clear zebra crossing detected."
            crossing_detected = False
            view_direction = "none"
        elif pred_class_idx == 0:  # Front-view
            status_text = "Possible zebra crossing detected ahead."
            crossing_detected = True
            view_direction = "ahead"
        elif pred_class_idx == 1:  # Left perspective
            status_text = "Possible zebra crossing detected on your left."
            crossing_detected = True
            view_direction = "left"
        elif pred_class_idx == 2:  # Right perspective
            status_text = "Possible zebra crossing detected on your right."
            crossing_detected = True
            view_direction = "right"
        else:
            status_text = "Crossing detection uncertain. Please verify your surroundings."
            crossing_detected = False
            view_direction = "uncertain"
            
        # 4. Create annotated output
        annotated = classical_res['annotated_stripes'].copy()
        
        # Optional walking-corridor overlay
        if include_path_awareness:
            annotated = generate_walkable_path_awareness(annotated)
            
        # Render prediction HUD badge
        badge_color = (0, 200, 0) if crossing_detected else (70, 70, 220)
        cv2.rectangle(annotated, (15, 15), (min(w - 15, 480), 95), (20, 20, 20), -1)
        cv2.rectangle(annotated, (15, 15), (min(w - 15, 480), 95), badge_color, 2)
        
        cv2.putText(annotated, f"Crosswalk: {pred_class_name}", (25, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Deep Conf: {pred_conf*100:.1f}% | CV Stripes: {stripes_count}", (25, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 230, 200), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Latency: {latency_ms:.1f}ms (CPU)", (25, 87),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        
        return {
            'predicted_class_idx': pred_class_idx,
            'predicted_class_name': pred_class_name,
            'confidence': round(pred_conf, 4),
            'class_probabilities': {self.class_names[i]: round(float(probs[i]), 4) for i in range(4)},
            'crossing_detected': crossing_detected,
            'view_direction': view_direction,
            'status_text': status_text,
            'stripe_confidence': stripe_conf,
            'candidate_stripes_count': stripes_count,
            'annotated_image': annotated,
            'latency_ms': round(latency_ms, 2),
            'pipeline_stages': classical_res['pipeline_stages']
        }
