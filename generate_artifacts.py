"""
Generate Sample Predictions, Figures, and Test Video for BlindSpot.
"""

import os
from pathlib import Path
os.environ['MPLCONFIGDIR'] = str(Path('.matplotlib').resolve())
Path('.matplotlib').mkdir(exist_ok=True)

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modules.obstacle_detection import ObstacleDetector
from modules.crosswalk_recognition import CrosswalkRecognizer
from modules.integration import BlindSpotSystem
from modules.image_processing import run_image_processing_pipeline

OUTPUT_PRED = Path('outputs/sample_predictions')
OUTPUT_FIGS = Path('outputs/figures')
OUTPUT_PRED.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGS.mkdir(parents=True, exist_ok=True)

print("Initializing system for artifact generation...")
system = BlindSpotSystem()

# 1. Process and save real sample predictions
test_samples = [
    ('dataset/crosswalk/Original/c1_p1_1.jpg', 'c1_front_crossing'),
    ('dataset/crosswalk/Original/c2_p1_1.jpg', 'c2_left_perspective'),
    ('dataset/crosswalk/Original/c3_p1_1.jpg', 'c3_right_perspective'),
    ('dataset/crosswalk/Original/c4_p1_2.jpg', 'c4_non_crosswalk')
]

for img_path, label in test_samples:
    p = Path(img_path)
    if p.exists():
        img = cv2.imread(str(p))
        res = system.process_frame(img)
        # Save composite output
        out_path = OUTPUT_PRED / f"integrated_{label}.jpg"
        cv2.imwrite(str(out_path), res['composite_image'])
        
        # Save crosswalk output
        cw_out_path = OUTPUT_PRED / f"crosswalk_{label}.jpg"
        cv2.imwrite(str(cw_out_path), res['crosswalk_annotated'])
        print(f"Saved predictions for {label}")

# 2. Generate DIP Pipeline Stage Grid Figure (Educational Demonstration)
sample_img = cv2.imread('dataset/crosswalk/Original/c1_p1_1.jpg')
dip_stages = run_image_processing_pipeline(sample_img)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
stages_to_plot = [
    ('Original BGR', cv2.cvtColor(dip_stages['original'], cv2.COLOR_BGR2RGB), None),
    ('Grayscale (Luminance)', dip_stages['grayscale'], 'gray'),
    ('Gaussian Filter (Noise Suppression)', dip_stages['blurred'], 'gray'),
    ("Otsu's Thresholding (Bimodal Contrast)", dip_stages['otsu_thresh'], 'gray'),
    ('Canny Edge Detection (Road ROI)', dip_stages['canny_roi'], 'gray'),
    ('Morphological Opening & Closing', dip_stages['morph_roi'], 'gray')
]

for idx, (title, img_data, cmap) in enumerate(stages_to_plot):
    r, c = divmod(idx, 3)
    ax = axes[r, c]
    if cmap:
        ax.imshow(img_data, cmap=cmap)
    else:
        ax.imshow(img_data)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.axis('off')

plt.suptitle('BlindSpot Digital Image Processing Pipeline Stages (DA3021)', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(OUTPUT_FIGS / 'dip_pipeline_stages.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved dip_pipeline_stages.png")

# 3. Generate a sample 3-second test video (10 fps, 30 frames) for video upload demonstration
video_path = Path("dataset/sample_test_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
sample_frame = cv2.imread('dataset/crosswalk/Original/c1_p1_1.jpg')
h, w = sample_frame.shape[:2]
vw = cv2.VideoWriter(str(video_path), fourcc, 10.0, (w, h))

# Simulate camera motion by panning slightly across frames
for f in range(30):
    shift_x = int(np.sin(f / 5.0) * 15)
    M = np.float32([[1, 0, shift_x], [0, 1, 0]])
    frame_panned = cv2.warpAffine(sample_frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    vw.write(frame_panned)

vw.release()
print(f"Generated sample test video at {video_path}")
