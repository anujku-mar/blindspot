"""
generate_slide_figures.py
Generates clean, high-resolution visual panels and figures for the PPT presentation.
Includes tested street photos and user-uploaded test photos.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIGS_DIR = Path("outputs/figures")
FIGS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR = Path("dataset/samples")

def create_dataset_overview_figure():
    """4-panel grid showing the 4 dataset classes trained on."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=200)
    plt.subplots_adjust(wspace=0.15, left=0.04, right=0.96, top=0.88, bottom=0.08)

    classes_info = [
        ("c1_p1_1.jpg", "Class 1: Front-View Crossing\n(150 images | Direct Approach)", "#1E88E5"),
        ("c2_p1_1.jpg", "Class 2: Left-Side Perspective\n(150 images | Angled Left)", "#F59E0B"),
        ("c3_p1_1.jpg", "Class 3: Right-Side Perspective\n(150 images | Angled Right)", "#8B5CF6"),
        ("c4_p1_2.jpg", "Class 4: Non-Crosswalk Asphalt\n(150 images | Negative Class)", "#EF4444"),
    ]

    for ax, (fname, label, col) in zip(axes, classes_info):
        p = SAMPLES_DIR / fname
        if p.exists():
            img = cv2.imread(str(p))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
        ax.set_title(label, fontsize=11, fontweight='bold', color=col, pad=8)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(col)
            spine.set_linewidth(3)

    plt.suptitle("Kaggle Crosswalk Dataset — 4 Perspective Classes (600 Total Images, 150/class)",
                 fontsize=14, fontweight='heavy', y=0.98, color='#0F172A')
    out_path = FIGS_DIR / "dataset_classes_grid.png"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Created: {out_path}")


def create_multimodal_test_grid():
    """4-panel grid of tested images demonstrating the system in action."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=200)
    plt.subplots_adjust(wspace=0.12, hspace=0.25, left=0.05, right=0.95, top=0.90, bottom=0.06)

    test_cases = [
        ("outputs/sample_predictions/integrated_c1_front_crossing.jpg",
         "Test Case 1: Crosswalk with Pedestrian Hazard (c1_p1_48.jpg)",
         "Perception: Person detected in RIGHT zone | Status: Front-View Crossing | Priority: MEDIUM"),
        ("outputs/sample_predictions/integrated_c2_left_perspective.jpg",
         "Test Case 2: Multi-Vehicle Street Hazard (c1_p1_146.jpg)",
         "Perception: Motorcycle & Cars in CENTRE path | Status: Crossing Detected | Priority: HIGH"),
        ("outputs/sample_predictions/user_real_test_1.jpg",
         "Test Case 3: Real Indoor / Hostel Obstacle Test (User Photo)",
         "Perception: Obstacle detected directly in CENTRE walking path | Alert: Caution!"),
        ("outputs/sample_predictions/integrated_c4_non_crosswalk.jpg",
         "Test Case 4: Open Asphalt Roadway (c4_p1_2.jpg)",
         "Perception: Roadway clear | Status: No clear zebra crossing detected | Priority: LOW"),
    ]

    for ax, (img_path, title, subtitle) in zip(axes.flat, test_cases):
        p = Path(img_path)
        if p.exists():
            img = cv2.imread(str(p))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=6, color='#1E3A8A')
        ax.set_xlabel(subtitle, fontsize=9.5, fontweight='semibold', color='#334155', labelpad=4)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("System Perception in Action Across Diverse Tested Scenes (Street & Real Indoor)",
                 fontsize=14, fontweight='heavy', y=0.98, color='#0F172A')
    out_path = FIGS_DIR / "tested_scenes_grid.png"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Created: {out_path}")


def create_architecture_flow_diagram():
    """Visual block diagram of the system pipeline."""
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=200)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    def draw_box(x, y, w, h, title, sub, color, text_color="white"):
        from matplotlib.patches import FancyBboxPatch
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", facecolor=color, edgecolor='#0F172A', linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', fontsize=10.5,
                fontweight='bold', color=text_color, zorder=3)
        ax.text(x + w/2, y + h*0.28, sub, ha='center', va='center', fontsize=8.5,
                color=text_color, zorder=3)

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#1E293B", lw=2, mutation_scale=15),
                    zorder=1)
        if label:
            ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.18, label, ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color='#475569')

    # Input
    draw_box(0.5, 2.3, 2.2, 1.4, "Visual Input", "Camera / Video Feed\n(RGB Frame)", "#3B82F6")
    draw_arrow(2.7, 3.0, 3.5, 3.0)

    # Preprocessing
    draw_box(3.5, 2.3, 2.2, 1.4, "OpenCV DIP", "Resize, Gaussian (5x5)\nCanny & Stripe ROI", "#0284C7")

    # Branch up & down
    draw_arrow(5.7, 3.5, 6.5, 4.3)
    draw_arrow(5.7, 2.5, 6.5, 1.7)

    # Upper Branch: Obstacle Detection
    draw_box(6.5, 3.6, 2.6, 1.4, "YOLOv8-nano", "Object Detection\n3-Zone Spatial Centroids", "#F59E0B")

    # Lower Branch: Crosswalk Recognition
    draw_box(6.5, 1.0, 2.6, 1.4, "MobileNetV3-Small", "4-Class Classifier\n+ DIP Stripe Verification", "#10B981")

    # Convergence
    draw_arrow(9.1, 4.3, 9.9, 3.5, "Spatial Zones")
    draw_arrow(9.1, 1.7, 9.9, 2.5, "Crossing State")

    # Decision Fusion
    draw_box(9.9, 2.3, 2.1, 1.4, "Decision Fusion", "Priority Arbitration\nCooldown Filter", "#8B5CF6")
    draw_arrow(12.0, 3.0, 12.6, 3.0)

    # Output
    draw_box(12.6, 2.3, 1.2, 1.4, "Spoken\nFeedback", "TTS Audio\n+ HUD", "#DC2626")

    plt.title("BlindSpot System Architecture & Multimodal Processing Pipeline",
              fontsize=13, fontweight='heavy', pad=12, color='#0F172A')
    out_path = FIGS_DIR / "system_architecture_diagram.png"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Created: {out_path}")

if __name__ == "__main__":
    create_dataset_overview_figure()
    create_multimodal_test_grid()
    create_architecture_flow_diagram()
    print("All slide figures generated successfully!")
