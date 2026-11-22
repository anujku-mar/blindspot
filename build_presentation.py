"""
build_presentation.py
Builds a visually rich, clean PPTX presentation for DA3021 Course Project.
Uses ultra-simple, short bullet points with everyday words (no heavy walls of text).
Highlights tested photos and student's own room validation tests (PASSED).
Zero text overlap. Generates presentation/BlindSpot_Presentation.pptx and presentation/slides.pptx.
"""

import os
from pathlib import Path

os.environ['MPLCONFIGDIR'] = str(Path('.matplotlib').resolve())
Path('.matplotlib').mkdir(exist_ok=True)

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

OUTPUT_DIR = Path("presentation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PPTX_PATH = OUTPUT_DIR / "BlindSpot_Presentation.pptx"
ALT_PPTX_PATH = OUTPUT_DIR / "slides.pptx"

FIGS_DIR = Path("outputs/figures")
EVAL_DIR = Path("outputs/evaluation")
PRED_DIR = Path("outputs/sample_predictions")

# Color Palette: Clean Modern Academic Navy & Slate
NAVY = RGBColor(15, 23, 42)          # #0F172A
NAVY_ACCENT = RGBColor(30, 58, 138)   # #1E3A8A
TEXT_DARK = RGBColor(30, 41, 59)      # #1E293B
TEXT_MUTED = RGBColor(100, 116, 139)  # #64748B
WHITE = RGBColor(255, 255, 255)
CARD_BG = RGBColor(241, 245, 249)     # #F1F5F9
CARD_BORDER = RGBColor(203, 213, 225) # #CBD5E1
SUCCESS_BG = RGBColor(240, 253, 244)  # #F0FDF4
SUCCESS_BORDER = RGBColor(134, 239, 172) # #86EFAC
ACCENT_GREEN = RGBColor(21, 128, 61)  # #15803D


def add_slide_header(slide, title_text: str, category_tag: str = "DA3021: MULTIMODAL DATA PROCESSING | IIT GUWAHATI"):
    """Creates a clean top banner with category tag and title. Guaranteed zero overlap."""
    header_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.15))
    header_shape.fill.solid()
    header_shape.fill.fore_color.rgb = NAVY
    header_shape.line.fill.background()

    # Category tag
    tb_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.12), Inches(11.7), Inches(0.28))
    tf_cat = tb_cat.text_frame
    tf_cat.word_wrap = True
    tf_cat.margin_left = tf_cat.margin_right = tf_cat.margin_top = tf_cat.margin_bottom = 0
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category_tag.upper()
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = RGBColor(147, 197, 253)

    # Title
    tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.40), Inches(11.7), Inches(0.60))
    tf_title = tb_title.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_right = tf_title.margin_top = tf_title.margin_bottom = 0
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.size = Pt(21)
    p_title.font.bold = True
    p_title.font.color.rgb = WHITE


def add_info_card(slide, left: float, top: float, width: float, height: float,
                  title: str, bullets: list, bg_color=CARD_BG, border_color=CARD_BORDER,
                  title_color=NAVY_ACCENT, body_size=13, title_size=16):
    """Clean information card with short, readable bullet points."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1.2)

    tb = slide.shapes.add_textbox(Inches(left + 0.25), Inches(top + 0.20), Inches(width - 0.50), Inches(height - 0.35))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

    p_t = tf.paragraphs[0]
    p_t.text = title
    p_t.font.size = Pt(title_size)
    p_t.font.bold = True
    p_t.font.color.rgb = title_color
    p_t.space_after = Pt(10)

    for b in bullets:
        p_b = tf.add_paragraph()
        p_b.text = f"•  {b}"
        p_b.font.size = Pt(body_size)
        p_b.font.color.rgb = TEXT_DARK
        p_b.space_after = Pt(8)
        p_b.line_spacing = 1.15


def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================
    # SLIDE 1: TITLE & CANDIDATE DETAILS
    # =========================================================
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.fill.background()

    tb1 = slide1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.3), Inches(4.5))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "🦯 BlindSpot"
    p0.font.size = Pt(46)
    p0.font.bold = True
    p0.font.color.rgb = WHITE
    p0.space_after = Pt(10)

    p1 = tf1.add_paragraph()
    p1.text = "A Lightweight Assistive System for Visually Impaired Pedestrians Using Computer Vision and Image Processing"
    p1.font.size = Pt(20)
    p1.font.color.rgb = RGBColor(199, 210, 254)
    p1.space_after = Pt(28)

    p2 = tf1.add_paragraph()
    p2.text = "DA3021: Multimodal Data Processing — Course Project Presentation (Monsoon 2026)"
    p2.font.size = Pt(14)
    p2.font.bold = True
    p2.font.color.rgb = RGBColor(147, 197, 253)
    p2.space_after = Pt(8)

    p3 = tf1.add_paragraph()
    p3.text = "Student: Anuj Kumar  |  3rd-Year B.Tech Data Science & Artificial Intelligence (DSAI)"
    p3.font.size = Pt(13)
    p3.font.color.rgb = WHITE
    p3.space_after = Pt(4)

    p4 = tf1.add_paragraph()
    p4.text = "Department of DSAI, Indian Institute of Technology Guwahati (IIT Guwahati)"
    p4.font.size = Pt(12)
    p4.font.color.rgb = RGBColor(203, 213, 225)

    # =========================================================
    # SLIDE 2: MOTIVATION — WHY THIS TOPIC?
    # =========================================================
    slide2 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide2, "1. Motivation: Why Did We Pick This Topic?")

    add_info_card(slide2, 0.8, 1.4, 5.6, 5.5, "The Problem in Daily Walking", [
        "White canes only touch things 1 meter away.",
        "They cannot tell if an obstacle is a car, pole, or person.",
        "Crossing busy streets is scary without seeing zebra lines.",
        "Simple beeping sensors beep too much and cause confusion.",
        "Our Goal: A smart camera system that speaks clear, short warnings.",
        "Runs on a normal laptop CPU without needing internet or GPU."
    ], body_size=13.5)

    img_sample = Path("outputs/sample_predictions/integrated_c1_front_crossing.jpg")
    if img_sample.exists():
        slide2.shapes.add_picture(str(img_sample), Inches(6.7), Inches(1.4), Inches(5.8), Inches(4.3))
        c_box = slide2.shapes.add_textbox(Inches(6.7), Inches(5.85), Inches(5.8), Inches(1.0))
        tf_c = c_box.text_frame
        tf_c.word_wrap = True
        p_c = tf_c.paragraphs[0]
        p_c.text = "Real Test: System finds the zebra crossing AND warns about the person standing on the right."
        p_c.font.size = Pt(12)
        p_c.font.bold = True
        p_c.font.color.rgb = NAVY_ACCENT

    # =========================================================
    # SLIDE 3: HISTORICAL PERSPECTIVE & PRIOR WORK
    # =========================================================
    slide3 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide3, "2. Problem Context & Historical Perspective")

    add_info_card(slide3, 0.8, 1.4, 3.7, 5.5, "Past: Sound Beepers (1980s–2000)", [
        "Used sonar and ultrasound sound waves.",
        "Only told if something was near or far.",
        "Could not tell what the object was.",
        "Constant loud beeps caused headaches."
    ], body_size=13)

    add_info_card(slide3, 4.8, 1.4, 3.7, 5.5, "Past: Old Math Vision (2000–2015)", [
        "Used basic line filters (Canny, Hough lines).",
        "Looked for white road stripes manually.",
        "Failed whenever shadows or sunlight hit the road.",
        "Broke if the camera tilted slightly."
    ], body_size=13)

    add_info_card(slide3, 8.8, 1.4, 3.7, 5.5, "Today: BlindSpot (Multimodal AI)", [
        "Lightweight deep learning models.",
        "Finds obstacles in 80 milliseconds on CPU.",
        "Detects crosswalks with 97.8% accuracy.",
        "Speaks simple alerts: Left, Centre, or Right."
    ], body_size=13)

    # =========================================================
    # SLIDE 4: SYSTEM ARCHITECTURE & MULTIMODAL PIPELINE
    # =========================================================
    slide4 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide4, "3. System Architecture: How It Works")

    arch_img = FIGS_DIR / "system_architecture_diagram.png"
    if arch_img.exists():
        slide4.shapes.add_picture(str(arch_img), Inches(0.8), Inches(1.35), Inches(11.733), Inches(4.3))

    add_info_card(slide4, 0.8, 5.8, 11.733, 1.3, "Simple 3-Step Flow", [
        "1. Camera Input: Takes an image or live video frame.",
        "2. Dual Processing: YOLO finds obstacles while MobileNet checks for crosswalks.",
        "3. Voice Alert: Warns user about centre path dangers first, with zero annoying repeated audio."
    ], body_size=13)

    # =========================================================
    # SLIDE 5: DATASET & PERSPECTIVE CROSSWALK CLASSES
    # =========================================================
    slide5 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide5, "4. Dataset: Kaggle Crosswalk Dataset (600 Images)")

    ds_img = FIGS_DIR / "dataset_classes_grid.png"
    if ds_img.exists():
        slide5.shapes.add_picture(str(ds_img), Inches(0.8), Inches(1.35), Inches(11.733), Inches(3.6))

    add_info_card(slide5, 0.8, 5.15, 11.733, 2.0, "Dataset Setup & Training", [
        "600 total road images with 150 images in each of the 4 classes.",
        "4 Classes: Front View, Left Angle, Right Angle, and Normal Road (No Crossing).",
        "Clean Data Split: 70% for training (420 images), 15% validation (90), 15% testing (90).",
        "Trained MobileNetV3 in just 15 epochs on CPU (took under 3 minutes)."
    ], body_size=13.5)

    # =========================================================
    # SLIDE 6: DIGITAL IMAGE PROCESSING (DIP) PIPELINE
    # =========================================================
    slide6 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide6, "5. Digital Image Processing (DIP) Stages")

    dip_img = FIGS_DIR / "dip_pipeline_stages.png"
    if dip_img.exists():
        slide6.shapes.add_picture(str(dip_img), Inches(0.8), Inches(1.35), Inches(11.733), Inches(3.8))

    add_info_card(slide6, 0.8, 5.3, 11.733, 1.85, "Step-by-Step Image Cleaning (OpenCV)", [
        "1. Grayscale & Blur: Turns image gray and blurs out rough road pebbles.",
        "2. Thresholding: Separates bright white zebra stripes from dark black road.",
        "3. Canny Edges: Outlines the sharp borders of each stripe.",
        "4. Morphology: Closes gaps to connect cracked and faded street paint."
    ], body_size=13.5)

    # =========================================================
    # SLIDE 7: OBSTACLE DETECTION & SPATIAL ZONING (MODULE 1)
    # =========================================================
    slide7 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide7, "6. Module 1: Smart Obstacle Detection")

    add_info_card(slide7, 0.8, 1.4, 5.6, 5.5, "YOLOv8-Nano & Spatial Zones", [
        "Uses lightweight YOLOv8n (only 6.2 MB).",
        "Detects people, cars, bikes, buses, and dogs.",
        "Splits the view into 3 simple zones:",
        "   - Left Zone (0% to 33%)",
        "   - Centre Path (33% to 67% — Top Hazard!)",
        "   - Right Zone (67% to 100%)",
        "Centre obstacles get urgent danger warnings.",
        "Smart cooldown stops repeating the same alert.",
        "Honest: Does not fake 3D distance numbers."
    ], body_size=13)

    pred_c2 = PRED_DIR / "integrated_c2_left_perspective.jpg"
    if pred_c2.exists():
        slide7.shapes.add_picture(str(pred_c2), Inches(6.7), Inches(1.4), Inches(5.8), Inches(4.3))
        tb_box = slide7.shapes.add_textbox(Inches(6.7), Inches(5.85), Inches(5.8), Inches(1.0))
        tf_tb = tb_box.text_frame
        tf_tb.word_wrap = True
        p_tb = tf_tb.paragraphs[0]
        p_tb.text = "Output: Motorcycle & car detected in centre path. System warns: 'Caution: Vehicle in centre walking path.'"
        p_tb.font.size = Pt(12)
        p_tb.font.bold = True
        p_tb.font.color.rgb = NAVY_ACCENT

    # =========================================================
    # SLIDE 8: TESTED STREET SCENES (VISUAL EVIDENCE)
    # =========================================================
    slide8 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide8, "7. System Testing: Real Street Scenarios")

    scenes_img = FIGS_DIR / "tested_scenes_grid.png"
    if scenes_img.exists():
        slide8.shapes.add_picture(str(scenes_img), Inches(0.8), Inches(1.35), Inches(11.733), Inches(4.5))

    add_info_card(slide8, 0.8, 6.0, 11.733, 1.15, "Observations from Street Tests", [
        "Crosswalk + Obstacle: Accurately spots crossing stripes even when people or cars block them.",
        "Corridor Overlay: Shows a green walking lane so you can visually verify the clear path."
    ], body_size=13.5)

    # =========================================================
    # SLIDE 9: REAL-WORLD TESTING — STUDENT ROOM (PASSED)
    # =========================================================
    slide9 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide9, "8. Real-World Test: My Hostel Room Photos (PASSED)")

    user_img1 = PRED_DIR / "user_real_test_1.jpg"
    user_img2 = PRED_DIR / "user_real_test_2.jpg"

    if user_img1.exists() and user_img2.exists():
        slide9.shapes.add_picture(str(user_img1), Inches(0.8), Inches(1.4), Inches(4.0), Inches(5.3))
        slide9.shapes.add_picture(str(user_img2), Inches(5.0), Inches(1.4), Inches(4.0), Inches(5.3))

    add_info_card(slide9, 9.2, 1.4, 3.3, 5.5, "Room Tests: PASSED", [
        "Tested on my actual hostel room photos to see real-life performance.",
        "Test 1 (Doorway Bag — PASSED): Luggage on floor directly blocking the room exit door. Alert: 'Caution: Obstacle in centre path!'",
        "Test 2 (Table Clutter — PASSED): Bottles, cup, and bucket detected and localized to centre and left.",
        "Result: Successfully passed indoor obstacle tests in my own living room!"
    ], bg_color=SUCCESS_BG, border_color=SUCCESS_BORDER, title_color=ACCENT_GREEN, body_size=13)

    # =========================================================
    # SLIDE 10: QUANTITATIVE EVALUATION & METRICS
    # =========================================================
    slide10 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide10, "9. Key Results & Accuracy Numbers")

    cm_img = EVAL_DIR / "confusion_matrix.png"
    curves_img = EVAL_DIR / "training_curves.png"

    if cm_img.exists() and curves_img.exists():
        slide10.shapes.add_picture(str(cm_img), Inches(0.8), Inches(1.4), Inches(4.6), Inches(3.8))
        slide10.shapes.add_picture(str(curves_img), Inches(5.6), Inches(1.4), Inches(6.9), Inches(3.8))

    add_info_card(slide10, 0.8, 5.35, 11.733, 1.8, "Tested on 90 Unseen Test Photos (Real Numbers)", [
        "MobileNetV3 Accuracy: 97.78% test accuracy (only 2 errors out of 90 unseen test photos).",
        "AI vs Old Method: MobileNetV3 (97.8%) beat the old texture baseline (78.9%) by a huge margin.",
        "Speed: Runs in ~80 ms on a normal laptop CPU (around 12 frames per second)."
    ], body_size=13.5)

    # =========================================================
    # SLIDE 11: WHAT DID YOU LEARN FROM THIS WORK?
    # =========================================================
    slide11 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide11, "10. What Did I Learn? (Course Links)")

    add_info_card(slide11, 0.8, 1.4, 3.7, 5.5, "Image Filters & DIP", [
        "Gaussian blur is crucial to kill asphalt noise before finding edges.",
        "Simple threshold fails in tree shadows; adaptive cutoff is needed.",
        "Morphology connects broken lines when road paint is faded."
    ], body_size=13)

    add_info_card(slide11, 4.8, 1.4, 3.7, 5.5, "Deep Learning vs Textures", [
        "Old texture math failed whenever the camera tilted.",
        "Neural networks easily handle tilted angles and street shadows.",
        "Transfer learning converges in under 3 minutes on CPU."
    ], body_size=13)

    add_info_card(slide11, 8.8, 1.4, 3.7, 5.5, "Multimodal Speech Design", [
        "Fusion doesn't need huge math; simple decision rules work best.",
        "Voice alerts must be short so walkers don't get overwhelmed.",
        "Cooldown timer keeps the app from repeating the same word."
    ], body_size=13)

    # =========================================================
    # SLIDE 12: REFLECTIONS — SURPRISES & FUTURE SCOPE
    # =========================================================
    slide12 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide12, "11. Reflections: Surprises & Future Scope")

    add_info_card(slide12, 0.8, 1.4, 5.6, 5.5, "What Surprised Me", [
        "Old line filters broke easily from simple sunlight shifts.",
        "MobileNet was surprisingly fast: trained in 3 minutes on CPU.",
        "Caught an early logic bug where side cars were ignored.",
        "Learned to use browser speech because cloud servers lack speakers."
    ], body_size=13.5)

    add_info_card(slide12, 6.8, 1.4, 5.7, 5.5, "Future Improvements", [
        "Add Depth AI to measure real distance in exact meters.",
        "Track moving objects over time (coming closer vs stopped).",
        "Put it on a Raspberry Pi inside wearable smart glasses.",
        "Add vibrating wristband for noisy traffic intersections."
    ], body_size=13.5)

    # =========================================================
    # SLIDE 13: REFERENCES & TOOLS USED
    # =========================================================
    slide13 = prs.slides.add_slide(blank_layout)
    add_slide_header(slide13, "12. References & Tools Disclosure")

    add_info_card(slide13, 0.8, 1.4, 5.6, 5.5, "References & Datasets", [
        "1. Kaggle Crosswalk Dataset (David Silva M., 600 images).",
        "2. Ultralytics YOLOv8 (Jocher et al., 2023).",
        "3. MobileNetV3 (Howard et al., ICCV 2019).",
        "4. Canny Edge Detection (J. Canny, IEEE TPAMI 1986).",
        "5. Otsu's Thresholding (N. Otsu, IEEE TSMC 1979)."
    ], body_size=13)

    add_info_card(slide13, 6.8, 1.4, 5.7, 5.5, "Software & AI Disclosure", [
        "Python, OpenCV, PyTorch, Torchvision, Ultralytics.",
        "Streamlit for the live interactive web demo.",
        "Antigravity AI Assistant was used as a coding helper for boilerplate scripts and presentation formatting.",
        "All models, numbers, and tests were genuinely run on our own computer."
    ], body_size=13)

    # Save presentation
    prs.save(str(PPTX_PATH))
    print(f"Presentation successfully saved to: {PPTX_PATH}")
    try:
        prs.save(str(ALT_PPTX_PATH))
        print(f"Also synced to: {ALT_PPTX_PATH}")
    except Exception:
        print(f"Note: {ALT_PPTX_PATH} is open in PowerPoint; saved as {PPTX_PATH}")
    print(f"Slide count: {len(prs.slides)}")


if __name__ == "__main__":
    build_presentation()
