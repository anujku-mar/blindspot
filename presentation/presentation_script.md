# BlindSpot: Spoken Presentation Script
**Course:** DA3021 Multimodal Data Processing (Monsoon 2026)  
**Institution:** Indian Institute of Technology Guwahati (IIT Guwahati)  
**Presenter:** [Your Name], 3rd-Year B.Tech in Data Science & Artificial Intelligence (DSAI)  
**Target Duration:** ~8.5 to 9.0 Minutes (Strictly Under the 10-Minute Course Limit)

---

## Slide 1: Title and Introduction (0:00 – 0:45)
"Hello everyone, respected instructor, and peers. 

My name is **[Your Name]**, and I am a 3rd-year B.Tech student in the Data Science and Artificial Intelligence (DSAI) branch here at IIT Guwahati. 

Today, I am presenting my course project for **DA3021: Multimodal Data Processing**, titled:
**'BlindSpot: A Lightweight Assistive System for Visually Impaired Pedestrians Using Computer Vision and Digital Image Processing.'**

The core motivation of this project is to explore how lightweight computer vision, digital image processing, and decision-level multimodal integration can come together to help visually impaired individuals better navigate outdoor pedestrian environments safely, using modest, edge-compatible compute resources."

---

## Slide 2: Motivation and Problem Statement (0:45 – 1:40)
"Why did I choose this topic? 

Worldwide, over 40 million individuals live with blindness, and nearly 300 million experience significant visual impairment. For these pedestrians, outdoor independent travel is fraught with danger. 

The traditional primary mobility aid is the white cane. The white cane is exceptional at providing mechanical tactile feedback for obstacles directly on the ground within arm's reach—about 1 to 1.2 meters. However, it offers zero look-ahead capability. A pedestrian cannot detect an approaching vehicle 10 meters away, nor can they determine whether they are properly aligned with a zebra crossing.

On the other hand, commercial electronic travel aids using ultrasonic or radar sensors tend to be frustratingly non-semantic: an ultrasonic cane emits the exact same high-frequency buzz whether you are walking toward a lamp post, a park bench, or a reversing truck. 

Our objective in BlindSpot was to create a lightweight, camera-based perception system that provides **semantic and spatial awareness**—telling the user *what* is in their environment, *where* it is spatially relative to their walking path, and whether there is an identifiable pedestrian crossing ahead, all communicated through clean, non-overloading spoken audio."

---

## Slide 3: Historical Perspective and Prior Work (1:40 – 2:35)
"To ground our work in the field, we looked at how assistive pedestrian navigation has evolved over the past four decades:

1. **Phase 1: Mechanical Mobility (Pre-1990s):** Dominated by the long white cane and guide dogs. Extremely reliable for tactile exploration, but blind to distant or dynamic traffic.
2. **Phase 2: Active Ranging Devices (1990s to 2010s):** Devices like the C-5 Laser Cane or ultrasonic bat sensors measured echo travel time. While innovative, they suffered from high false alarm rates on curbs and caused cognitive fatigue through constant, annoying beeping.
3. **Phase 3: Classical Computer Vision (2010 to 2018):** Researchers began using early edge detectors, Hough transforms, and bipolar intensity profiling to detect parallel zebra stripes. However, these systems were notoriously brittle under tree shadows, road puddles, and camera tilt.
4. **Phase 4: Modern Multimodal Assistive Vision (2018 to Present):** Today, with lightweight deep learning, edge devices can perform both object detection and classification in real time. 

BlindSpot builds upon this modern paradigm. We deliberately combine deep learning with classical digital image processing, bridging them through an explainable, decision-level multimodal fusion layer."

---

## Slide 4: Proposed BlindSpot Architecture (2:35 – 3:30)
"Here is the system architecture of BlindSpot:

The pipeline starts with a camera image or video frame at 720p. 

After initial preprocessing and road Region of Interest extraction, the pipeline splits into two independent perception streams:
1. **Stream 1: Obstacle Detection**, powered by a pretrained YOLOv8-nano model, which identifies relevant pedestrian entities and projects their 2D bounding-box centroids into three horizontal spatial zones: **Left**, **Centre**, and **Right**.
2. **Stream 2: Zebra Crossing Recognition**, which combines a fine-tuned MobileNetV3-Small neural network with classical morphological stripe-periodicity verification.

Both streams feed directly into our **Decision-Level Alert Layer**. 
Instead of relying on a black-box end-to-end multimodal network, we designed an explicit rule-based decision arbiter. This ensures that:
- Hazards in the central forward trajectory take immediate priority.
- Crossing notifications are spoken clearly.
- A 3-second cooldown mechanism suppresses repetitive audio clutter.

Finally, the synthesized alert is converted into spoken audio using modern speech synthesis, alongside an accessibility-focused visual HUD."

---

## Slide 5: Dataset and Preprocessing (3:30 – 4:25)
"For our crosswalk module, we utilized the public **Kaggle Crosswalk Dataset** by David Silva. 

Upon downloading and inspecting the archive, we verified that it contains exactly 600 high-resolution JPEG images (1280x720) across four perfectly balanced classes—150 images each:
- **Class 1:** Front-view crossing
- **Class 2:** Left-side perspective
- **Class 3:** Right-side perspective
- **Class 4:** Non-crosswalk road scenes

To adhere strictly to machine learning best practices and prevent data leakage, we implemented a stratified split:
- **70% Training Set (420 images)**
- **15% Validation Set (90 images)**
- **15% Independent Test Set (90 images)**

During training, we applied light data augmentations—moderate random horizontal flipping and subtle color jittering—to build robustness against varying outdoor illumination while preserving true geometric perspectives."

---

## Slide 6: Application 1 — Smart Obstacle Detection (4:25 – 5:20)
"Let us look at Application 1: Smart Obstacle Detection.

We selected **YOLOv8-nano** (`yolov8n.pt`), which is only 6.2 megabytes and has 3.2 million parameters. 

From the 80 COCO classes, we filtered out non-essential objects and focused strictly on pedestrian-critical entities: `person`, `car`, `bicycle`, `motorcycle`, `bus`, `truck`, `dog`, and `traffic light`.

The key innovation here is our **3-zone horizontal spatial partitioning**:
- The left 33% of the image corresponds to the **Left Zone**.
- The middle 34% corresponds to the **Centre Zone** (the pedestrian's forward walking path).
- The right 33% corresponds to the **Right Zone**.

When an object is detected, we calculate its bounding-box centroid:
`Cx = (x1 + x2) / 2`
and classify its zone. 

Crucially, as a responsible engineering decision, **we explicitly do not claim distance estimation or collision prediction**. Estimating true metric depth from a single 2D camera without validated stereo or LiDAR is scientifically unreliable. Instead, we formulate qualitative spatial warnings like:
*'Caution: Vehicle detected in your centre path'* or *'Person detected on your left.'*"

---

## Slide 7: Application 2 — Zebra Crossing Recognition & DIP Pipeline (5:20 – 6:15)
"Application 2 is Zebra Crossing Recognition and Walking Path Awareness.

Here, we deeply integrated core concepts from our DA3021 course:
1. **Luminance Extraction:** Converting the BGR scene to Grayscale.
2. **Gaussian Smoothing:** We applied a 5x5 Gaussian filter with sigma=1.4 to suppress asphalt grain and gravel noise without blurring stripe boundaries.
3. **Thresholding:** We implemented both Otsu's bimodal thresholding and Adaptive Gaussian thresholding to handle harsh sunlight and tree shadows.
4. **Canny Edge Detection:** With hysteresis thresholds 50 and 150, restricted to a trapezoidal road Region of Interest.
5. **Morphological Filtering:** We designed custom rectangular structuring elements—a 5x5 morphological close to bridge cracks and asphalt wear, followed by a 9x3 horizontal open to isolate rectangular stripe contours.
6. **Stripe Periodicity & Aspect Ratio:** We checked that detected stripes conform to regular aspect ratios between 0.4 and 18.0 and exhibit periodic spacing along the road axis.

Simultaneously, our fine-tuned MobileNetV3-Small classifier categorizes the scene into Front, Left, Right, or Non-Crosswalk, enabling safe spoken advice such as: *'Possible zebra crossing detected ahead.'*"

---

## Slide 8: Results and Quantitative Evaluation (6:15 – 7:15)
"Now, let us examine our genuine, measured experimental results. In strict adherence to academic honesty, **not a single metric here is fabricated**.

On our held-out 90-image test set:
- **Overall Classification Accuracy:** **97.78%**
- **Macro Precision:** **97.82%**
- **Macro Recall:** **97.78%**
- **Macro F1-Score:** **97.78%**

As seen in the confusion matrix on Slide 8:
- Front-view crossing achieved 96% recall and 100% precision.
- Left-side perspective achieved a perfect 100% precision and recall.
- Right-side perspective achieved 100% recall and 96% precision.
- Non-crosswalk achieved 95% precision and 95% recall.

Furthermore, we benchmarked our deep model against a **Classical GLCM Texture Baseline** using 10 Gray-Level Co-occurrence Matrix features (ASM, contrast, entropy, homogeneity) paired with a Random Forest classifier. 
The GLCM baseline achieved **78.89% accuracy**. 
Our fine-tuned MobileNetV3 achieved **97.78%**—a dramatic improvement of nearly **19 percentage points**. This demonstrates that deep convolutional filters capture spatial perspective and geometric regularity far better than second-order statistical texture descriptors.

In terms of execution speed on a standard CPU:
- MobileNetV3 takes just **43 milliseconds** per frame (~23 FPS).
- YOLOv8-nano takes **75 milliseconds** (~13 FPS).
- The full multimodal pipeline runs at **~8.5 to 9 FPS on a standard laptop CPU**, which is more than fast enough for walking-speed pedestrian assistance."

---

## Slide 9: Course Syllabus Connection (7:15 – 7:55)
"Slide 9 illustrates how BlindSpot directly bridges theory and practice across the DA3021 syllabus:

- **Digital Filtering:** 5x5 Gaussian kernels for noise suppression in `image_processing.py`.
- **Edge Detection:** Canny hysteresis edge detection in `image_processing.py`.
- **Thresholding:** Otsu and Adaptive Gaussian bimodal segmentation for road scenes.
- **Morphological Operations:** Custom rectangular structuring elements for bridging stripe fissures.
- **Feature Extraction:** Comparative evaluation of GLCM texture features against deep CNN representations.
- **Deep Classification:** Fine-tuning MobileNetV3-Small with cross-entropy loss and cosine annealing.
- **Real-Time Detection:** YOLOv8n object localization and spatial zoning in `obstacle_detection.py`.
- **Multimodal Fusion:** Decision-level fusion and text-to-speech synthesis in `integration.py`."

---

## Slide 10: Reflections, Surprises, Limitations & Safety (7:55 – 8:35)
"What did I learn, and what surprised me during this project?

First, I was surprised by how brittle classical thresholding techniques can be in real street scenes. While Otsu's thresholding works beautifully on textbook images with clear bimodal histograms, outdoor tree shadows split the road histogram into three or four peaks, causing Otsu to fail. Combining adaptive thresholding with a deep neural classifier was essential to achieve robustness.

Second, I learned that in multimodal assistive systems, **audio ergonomics are as vital as computer vision accuracy**. In initial tests, speaking every detection in every frame produced an overwhelming, unusable cacophony. Implementing a 3-second cooldown and capping simultaneous spoken warnings to high-priority items made the system genuinely usable.

Finally, we maintain a strict ethical boundary:
BlindSpot is a research prototype designed to augment awareness, **never to replace a white cane or guide dog**. Our system never says *'The road is safe'* or *'You can cross now'*. It uses cautious phrasing: *'Possible zebra crossing detected. Please verify your surroundings.'*"

---

## Slide 11: References & Future Scope (8:35 – 9:00)
"Looking to the future, we envision three major extensions:
1. Integrating lightweight monocular depth estimation networks (like FastDepth) to provide validated metric proximity.
2. Porting the pipeline to wearable smart-glasses powered by a Raspberry Pi 5 with an Edge TPU accelerator.
3. Adding traffic signal state classification (detecting red hand vs. green walking man signals).

In conclusion, BlindSpot demonstrates that lightweight computer vision and classical image processing can be fused effectively on low-power hardware to deliver practical, meaningful assistance for visually impaired pedestrians.

Thank you very much for your time and attention. I am happy to address any questions."
