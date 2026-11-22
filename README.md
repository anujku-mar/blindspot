# BlindSpot: A Lightweight Assistive System for Visually Impaired Pedestrians

**Course Project:** DA3021: Multimodal Data Processing (Monsoon 2026)  
**Institution:** Indian Institute of Technology Guwahati (IIT Guwahati)  
**Academic Program:** 3rd-Year B.Tech in Data Science & Artificial Intelligence (DSAI)  
**Author:** [Your Name]  
**Project Repository:** BlindSpot

---

## 1. Project Overview & Motivation

Independent outdoor navigation is one of the most significant daily hurdles faced by visually impaired individuals worldwide. Traditional mobility aids such as the long white cane provide tactile feedback for immediate ground obstacles at physical contact distance (< 1.2 m), but offer zero look-ahead capability to anticipate dynamic traffic or identify pedestrian crossings at a distance. Electronic Travel Aids (ETAs) that rely solely on ultrasonic distance sensors produce generic acoustic beeps without semantic or spatial context.

**BlindSpot** is a lightweight, edge-compatible assistive system that bridges this gap by unifying:
1. **Spatial Object Detection:** Identifying pedestrians, vehicles, bicycles, and hazards, partitioned into **Left**, **Centre**, and **Right** spatial zones using a pretrained YOLOv8-nano model.
2. **Hybrid Zebra Crossing Recognition:** Classifying pedestrian crossing orientation (Front, Left, Right, None) using a fine-tuned MobileNetV3-Small classifier combined with classical Digital Image Processing (DIP) stripe-periodicity verification.
3. **Decision-Level Multimodal Integration:** Merging dual visual streams into clear, non-overloading spoken acoustic warnings with built-in deduplication and safety cooldowns.

---

## 2. System Architecture & Multimodal Pipeline

BlindSpot operates via an explainable, decision-level multimodal fusion pipeline:

```
Camera Input (Image / Video Stream)
              │
              ▼
    Image Preprocessing & ROI
              │
      ┌───────┴─────────────────────────┐
      ▼                                 ▼
Obstacle Detection             Crosswalk Recognition
 (YOLOv8-nano)                  (MobileNetV3-Small + DIP)
      │                                 │
      ▼                                 ▼
Object Location & Category      Perspective & Stripe State
(Left / Centre / Right Zones)   (Front / Left / Right / None)
      │                                 │
      └───────┬─────────────────────────┘
              ▼
   Decision-Level Fusion & Alert Arbiter
              │
              ▼
   Multimodal Spoken Audio Feedback (Web Speech / gTTS)
```

---

## 3. Data Sources & Verification

### Dataset 1: Kaggle Crosswalk Dataset
- **Source Link:** [Kaggle Crosswalk Dataset (davidsilvam/crosswalkdataset)](https://www.kaggle.com/datasets/davidsilvam/crosswalkdataset)
- **Actual Download Size:** **~136 MB** (compressed archive), containing exactly **600 high-resolution JPEG images** (1280x720).
- **Integrity Check:** Automated inspection verified **600 valid images, 0 corrupted samples**.
- **Class Distribution (150 images per class):**
  - `c1`: Front-view crossing (Direct pedestrian crossing path)
  - `c2`: Left-side perspective crossing
  - `c3`: Right-side perspective crossing
  - `c4`: Non-crosswalk road scenes
- **Dataset Manifest:** Formally documented in [`dataset/dataset_manifest.json`](dataset/dataset_manifest.json).
- **Stratified Train / Val / Test Split (Zero Data Leakage):**
  - **Training Set (70%):** 420 images (105 per class)
  - **Validation Set (15%):** 90 images (strictly held-out tuning)
  - **Test Set (15%):** 90 images (strictly unseen evaluation)

### Model 2: COCO-Pretrained YOLOv8-nano
- **Source:** Ultralytics YOLOv8-nano (`yolov8n.pt`, 6.2 MB, 3.2M parameters)
- **COCO Pedestrian Filter:** 8 safety-critical classes: `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`, `dog`, `traffic light`.
- **Pretrained Weights:** Stored in [`models/pretrained_weights/yolov8n.pt`](models/pretrained_weights/yolov8n.pt).

---

## 4. Digital Image Processing (DIP) Pipeline

In compliance with the **DA3021: Multimodal Data Processing** syllabus, BlindSpot implements a comprehensive image processing pipeline in [`modules/image_processing.py`](modules/image_processing.py):

1. **Luminance Extraction:** Grayscale conversion (`cv2.cvtColor`).
2. **Noise Reduction:** 5x5 Gaussian filter ($\sigma=1.4$) suppressing asphalt aggregate noise while preserving crisp stripe transitions.
3. **Bimodal & Adaptive Thresholding:**
   - Otsu's global thresholding for high-contrast road illumination.
   - Adaptive Gaussian thresholding ($15\times 15$ window, $C=3$) for road regions under severe building and canopy shadows.
4. **Canny Edge Detection:** Double-threshold hysteresis (50/150) bounded by a forward trapezoidal road Region of Interest (ROI).
5. **Morphological Filtering:**
   - Morphological Close (5x5 rectangular kernel) to bridge asphalt wear fissures.
   - Morphological Open (9x3 horizontal kernel) to eliminate non-stripe speckle noise.
6. **Stripe Periodicity & Aspect Ratio Verification:** Filters candidate stripe contours with aspect ratios between 0.4 and 18.0 and computes inter-stripe vertical variance to confirm regular zebra stripe cadence.
7. **Walking Corridor Visual Awareness:** Shaded forward trajectory polygon providing approximate visual guidance without claiming collision avoidance.

---

## 5. Measured Experimental Results (Genuine Benchmarking)

In accordance with strict academic guidelines, **all reported metrics are genuinely measured from our experimental runs**:

### Crosswalk Classifier Test Set Metrics (Held-Out 90 Images)
- **Model Architecture:** Fine-Tuned MobileNetV3-Small (15 epochs, AdamW optimizer, cosine annealing schedule)
- **Overall Accuracy:** **97.78%**
- **Macro Precision:** **97.82%**
- **Macro Recall:** **97.78%**
- **Macro F1-Score:** **97.78%**
- **Average CPU Inference Latency:** **43.03 ms** per frame (**~23.2 FPS** on pure CPU)

#### Detailed Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Front-View Crossing** | 1.0000 | 0.9565 | 0.9778 | 23 |
| **Left-Side Perspective** | 1.0000 | 1.0000 | 1.0000 | 22 |
| **Right-Side Perspective** | 0.9583 | 1.0000 | 0.9787 | 23 |
| **Non-Crosswalk** | 0.9545 | 0.9545 | 0.9545 | 22 |

#### Baseline Comparison: Deep CNN vs Classical GLCM Texture Descriptors
- **MobileNetV3-Small (Deep Features):** **97.78%** Test Accuracy | **97.78%** Macro F1
- **GLCM Texture Features + Random Forest (10 statistical features):** **78.89%** Test Accuracy | **78.99%** Macro F1
- **Observation:** Deep convolutional representations yield a **+18.89% accuracy boost** over Gray-Level Co-occurrence Matrix (GLCM) features because severe road perspective distortions degrade static angle-dependent co-occurrence matrices.

### Obstacle Detection Benchmark
- **Model:** YOLOv8-nano (CPU execution)
- **Inference Latency:** **75–85 ms** per image (**~12–13 FPS** on modern multi-core CPU)
- **Full Multimodal Pipeline Latency:** **~118–125 ms** (**~8–9 FPS** on pure CPU)

---

## 6. DA3021 Course Syllabus Connection

| Course Topic | Where It Is Implemented | Technical Description |
| :--- | :--- | :--- |
| **Digital Filtering & Smoothing** | `modules/image_processing.py` | 5x5 Gaussian kernel ($\sigma=1.4$) to eliminate pavement grain while retaining stripe boundaries. |
| **Edge Detection & Boundaries** | `modules/image_processing.py` | Canny edge detection with 50/150 hysteresis thresholds bounded by a road trapezoidal ROI. |
| **Thresholding & Segmentation** | `modules/image_processing.py` | Otsu's bimodal thresholding and Adaptive Gaussian thresholding for dynamic sunlight and shade. |
| **Morphological Operations** | `modules/image_processing.py` | Rectangular closing (5x5) for asphalt fissures + horizontal opening (9x3) for stripe isolation. |
| **Feature Extraction & Textures** | `train_crosswalk.py` | Comparative evaluation of 10 GLCM texture statistical descriptors against CNN feature maps. |
| **Lightweight Deep Classification** | `modules/crosswalk_recognition.py` | MobileNetV3-Small fine-tuned for 4-class perspective classification on CPU. |
| **Spatial Object Detection** | `modules/obstacle_detection.py` | Pretrained YOLOv8n with 3-way horizontal centroid partitioning (Left, Centre, Right). |
| **Multimodal Integration & Speech** | `modules/integration.py` | Decision-level alert arbitration with Web Speech API and gTTS audio synthesis. |

---

## 7. Project Structure

```
BlindSpot/
│
├── app.py                      # Interactive Streamlit Web Application
├── config.py                   # Central paths, thresholds, and spatial zone boundaries
├── requirements.txt            # Python dependencies
├── README.md                   # Complete technical documentation
├── train_crosswalk.py          # Model training & GLCM baseline script
├── evaluate_and_save.py        # Independent test set evaluation & plotting script
├── generate_artifacts.py       # Sample predictions, figures & test video generator
├── build_presentation.py       # Programmatic PPTX and PDF slides generator
│
├── modules/
│   ├── __init__.py
│   ├── obstacle_detection.py   # YOLOv8n inference with 3-zone horizontal spatial zoning
│   ├── crosswalk_recognition.py# MobileNetV3 classifier + OpenCV stripe periodicity
│   ├── image_processing.py     # Classical DIP pipeline (Gaussian, Otsu, Canny, Morphology)
│   ├── voice_assistance.py     # Text-to-speech engine with Web Speech & gTTS fallbacks
│   └── integration.py          # Decision-level multimodal perception fusion layer
│
├── dataset/
│   ├── dataset_manifest.json   # Full manifest of verified Kaggle crosswalk dataset
│   ├── sample_test_video.mp4   # 3-second motion video for video upload demonstration
│   └── crosswalk/
│       ├── crosswalk-dataset-description.txt
│       ├── 10_FEATURES_M17_CM6b_TH199.csv
│       └── Original/           # 600 verified JPEG images (c1, c2, c3, c4)
│
├── models/
│   ├── crosswalk_classifier.pth# Trained MobileNetV3-Small model weights
│   └── pretrained_weights/
│       └── yolov8n.pt          # Pretrained YOLOv8-nano weights
│
├── outputs/
│   ├── evaluation/
│   │   ├── crosswalk_metrics.json  # Genuine JSON evaluation metrics
│   │   ├── confusion_matrix.png    # 4x4 test set confusion matrix plot
│   │   ├── training_curves.png     # Loss and accuracy vs epoch curves
│   │   ├── train_split.csv         # Stratified training split file list
│   │   ├── val_split.csv           # Stratified validation split file list
│   │   └── test_split.csv          # Stratified test split file list
│   │
│   ├── figures/
│   │   └── dip_pipeline_stages.png # 6-stage DIP educational visual inspector
│   │
│   └── sample_predictions/     # Working prediction outputs across classes
│       ├── integrated_c1_front_crossing.jpg
│       ├── integrated_c2_left_perspective.jpg
│       ├── integrated_c3_right_perspective.jpg
│       ├── integrated_c4_non_crosswalk.jpg
│       └── crosswalk_c1_front_crossing.jpg
│
├── tests/
│   ├── test_obstacle.py        # Unit tests for obstacle detection & spatial zoning
│   ├── test_crosswalk.py       # Unit tests for crosswalk recognition & DIP filters
│   └── test_integration.py     # End-to-end integration tests & audio verification
│
└── presentation/
    ├── slides.pptx             # 11-slide presentation in Microsoft PowerPoint format
    ├── slides.pdf              # 11-slide presentation in PDF format (DA3021 submission deliverable)
    └── presentation_script.md  # Spoken presentation script (timed for 8.5–9.0 minutes)
```

---

## 8. Safety & Ethical Disclaimers

> [!WARNING]
> **Research & Educational Prototype Notice:**
> - BlindSpot is an academic research prototype built for educational demonstration.
> - **It does NOT replace** the white cane, guide dog, or human navigation assistance.
> - The system **never claims metric distance estimation or collision avoidance** from a single monocular 2D camera.
> - The system **never says**: *"The road is safe"*, *"You can cross now"*, or *"There are no obstacles"*.
> - All synthesized speech alerts use cautious probabilistic language: *"Possible zebra crossing detected ahead. Please verify your surroundings."*

---

## 9. Installation & Execution Guide

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 (Tested on Python 3.13 64-bit on Windows)
- Modern web browser (Chrome, Edge, or Firefox)

### Step 1: Clone or Navigate to Directory
```bash
cd "c:\Users\Anuj Kumar\OneDrive\Desktop\Multimodal_project"
```

### Step 2: Set Up Virtual Environment & Install Dependencies
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Run Automated Test Suite
Verify that all 15 unit and integration tests pass:
```bash
.\.venv\Scripts\python -m pytest -v tests/
```

### Step 4: Launch the Interactive Streamlit Application
```bash
.\.venv\Scripts\streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Step 5: (Optional) Re-run Evaluation or Re-generate Presentation
```bash
# Re-run evaluation and generate metrics & figures:
.\.venv\Scripts\python evaluate_and_save.py

# Re-build presentation PPTX and PDF slides:
.\.venv\Scripts\python build_presentation.py
```

---

## 10. Reflections & Future Work

1. **Monocular Depth Integration:** Single-camera 2D bounding boxes do not yield true Euclidean metric distance. Future iterations will integrate lightweight monocular depth networks (e.g., FastDepth) or stereo vision.
2. **Edge Hardware Acceleration:** Porting the models onto a wearable Raspberry Pi 5 paired with a Coral Edge TPU or Intel Neural Compute Stick for sub-20 ms latency.
3. **Traffic Light Phase Recognition:** Expanding the detection filter to specifically identify pedestrian signal heads (Walk / Do Not Walk icons).
