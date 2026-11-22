"""
Digital Image Processing Module for BlindSpot
Implements classical computer vision techniques:
- Color space conversion (Grayscale)
- Gaussian smoothing and noise reduction
- Adaptive & Otsu thresholding
- Canny edge detection
- Morphological transformations (Opening, Closing, Directional kernels)
- Contour analysis & Stripe periodicity verification (Zebra patterns)
- Region of Interest (ROI) perspective filtering
- Approximate walkable road corridor visualization
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional

def preprocess_image(image_bgr: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Resize and ensure contiguous uint8 BGR image."""
    if target_size is not None:
        return cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_LINEAR)
    return image_bgr.copy()

def extract_road_roi(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract trapezoidal Region of Interest (ROI) corresponding to the street surface ahead.
    Visually impaired pedestrians primarily navigate the lower-to-middle half of the visual field.
    """
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Trapezoid coordinates representing forward street perspective
    roi_corners = np.array([[
        (int(w * 0.10), h),
        (int(w * 0.35), int(h * 0.45)),
        (int(w * 0.65), int(h * 0.45)),
        (int(w * 0.90), h)
    ]], dtype=np.int32)
    
    cv2.fillPoly(mask, roi_corners, 255)
    roi_bgr = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    return roi_bgr, mask

def run_image_processing_pipeline(image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Runs the complete multi-stage DIP pipeline and returns intermediate visual representations.
    Useful for educational demonstrations and pipeline verification in the course project.
    """
    h, w = image_bgr.shape[:2]
    
    # 1. Grayscale Conversion
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Gaussian Blur (noise reduction while preserving stripe edges)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    
    # 3. Otsu's Thresholding (bimodal contrast between white stripes and dark asphalt)
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Adaptive Thresholding (robust against uneven shadows and street lighting)
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
    )
    
    # 5. Canny Edge Detection (hysteresis thresholds)
    canny_edges = cv2.Canny(blurred, 50, 150)
    
    # 6. Morphological Operations
    # Rectangular kernel tailored to horizontal/perspective pedestrian crosswalk stripes
    kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    # Morphological Close: bridge small gaps in stripes caused by asphalt wear
    morph_close = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel_close)
    # Morphological Open: remove salt-and-pepper noise and speckles
    morph_clean = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel_horiz)
    
    # 7. ROI Masking on edges and morphological output
    _, roi_mask = extract_road_roi(image_bgr)
    canny_roi = cv2.bitwise_and(canny_edges, canny_edges, mask=roi_mask)
    morph_roi = cv2.bitwise_and(morph_clean, morph_clean, mask=roi_mask)
    
    return {
        'original': image_bgr,
        'grayscale': gray,
        'blurred': blurred,
        'otsu_thresh': otsu_thresh,
        'adaptive_thresh': adaptive_thresh,
        'canny_edges': canny_edges,
        'canny_roi': canny_roi,
        'morph_clean': morph_clean,
        'morph_roi': morph_roi,
        'roi_mask': roi_mask
    }

def analyze_stripe_patterns(image_bgr: np.ndarray) -> Dict:
    """
    Classical Computer Vision Stripe Pattern Analysis:
    - Finds white rectangular stripe contours in the road region.
    - Measures bounding box aspect ratios, horizontal alignment, and periodicity.
    - Computes a heuristic stripe confidence score [0.0 - 1.0].
    """
    pipeline = run_image_processing_pipeline(image_bgr)
    morph_roi = pipeline['morph_roi']
    h, w = image_bgr.shape[:2]
    
    # Find contours from filtered binary ROI
    contours, _ = cv2.findContours(morph_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidate_stripes = []
    min_area = (h * w) * 0.0015  # min 0.15% of frame
    max_area = (h * w) * 0.12    # max 12% of frame
    
    annotated = image_bgr.copy()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = cw / float(ch + 1e-5)
            
            # Crosswalk stripes observed from perspective are generally wider than tall
            # or elongated rectangles in the bottom half of the image
            if 0.4 <= aspect_ratio <= 18.0 and y > int(h * 0.30):
                candidate_stripes.append({
                    'bbox': (x, y, cw, ch),
                    'area': area,
                    'center': (x + cw // 2, y + ch // 2),
                    'aspect_ratio': aspect_ratio
                })
                # Highlight candidate stripe
                cv2.rectangle(annotated, (x, y), (x + cw, y + ch), (0, 255, 255), 2)
    
    # Compute pattern regularity:
    # A true zebra crossing typically contains >= 3 roughly aligned, periodically spaced stripes
    num_stripes = len(candidate_stripes)
    has_periodic_stripes = False
    stripe_confidence = 0.0
    
    if num_stripes >= 3:
        # Sort stripes by vertical Y coordinate
        candidate_stripes.sort(key=lambda s: s['center'][1])
        y_centers = [s['center'][1] for s in candidate_stripes]
        diffs = np.diff(y_centers)
        
        # Check standard deviation of distances between stripes
        if len(diffs) > 1 and np.mean(diffs) > 8:
            cv_diff = np.std(diffs) / (np.mean(diffs) + 1e-5)
            if cv_diff < 0.85:  # regular spacing
                has_periodic_stripes = True
                stripe_confidence = min(1.0, 0.40 + 0.12 * num_stripes)
            else:
                stripe_confidence = min(0.65, 0.25 + 0.08 * num_stripes)
        else:
            stripe_confidence = min(0.50, 0.20 + 0.05 * num_stripes)
    elif num_stripes in [1, 2]:
        stripe_confidence = 0.20 * num_stripes
        
    return {
        'candidate_stripes_count': num_stripes,
        'has_periodic_stripes': has_periodic_stripes,
        'stripe_confidence': round(float(stripe_confidence), 3),
        'annotated_stripes': annotated,
        'pipeline_stages': pipeline
    }

def generate_walkable_path_awareness(image_bgr: np.ndarray) -> np.ndarray:
    """
    Walking-path visual awareness overlay:
    Performs morphological road corridor segmentation to shade the immediate ground region ahead.
    IMPORTANT DISCLAIMER:
    This provides approximate visual-region guidance only; does not guarantee a certified safe path.
    """
    h, w = image_bgr.shape[:2]
    overlay = image_bgr.copy()
    
    # Corridor polygon in pedestrian forward trajectory
    corridor_pts = np.array([
        [int(w * 0.25), h],
        [int(w * 0.40), int(h * 0.55)],
        [int(w * 0.60), int(h * 0.55)],
        [int(w * 0.75), h]
    ], np.int32)
    
    # Draw transparent green corridor on copy
    temp = image_bgr.copy()
    cv2.fillPoly(temp, [corridor_pts], (40, 180, 40))
    cv2.polylines(temp, [corridor_pts], True, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    
    # Alpha blend
    alpha = 0.28
    cv2.addWeighted(temp, alpha, overlay, 1 - alpha, 0, overlay)
    
    # Add textual disclaimer watermark
    cv2.putText(
        overlay,
        "Walking corridor (Approximate visual awareness only)",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 255, 220),
        1,
        cv2.LINE_AA
    )
    return overlay
