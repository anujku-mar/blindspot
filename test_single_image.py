"""
Quick CLI Test Tool for BlindSpot
Usage:
    python test_single_image.py
    python test_single_image.py --image dataset/crosswalk/Original/c1_p1_10.jpg
    python test_single_image.py --image "C:/path/to/your/custom_image.jpg" --speak
"""

import argparse
import sys
import os
from pathlib import Path

# Setup paths
os.environ['YOLO_CONFIG_DIR'] = str(Path('.ultralytics').resolve())
os.environ['MPLCONFIGDIR'] = str(Path('.matplotlib').resolve())
Path('.ultralytics').mkdir(exist_ok=True)
Path('.matplotlib').mkdir(exist_ok=True)

import cv2
from modules.integration import BlindSpotSystem
from modules.voice_assistance import speak_offline_pyttsx3

def main():
    parser = argparse.ArgumentParser(description="Test BlindSpot on a single image and hear/save audio alert.")
    parser.add_argument(
        "--image",
        type=str,
        default="dataset/crosswalk/Original/c1_p1_48.jpg",
        help="Path to the test image file (JPG/PNG)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/sample_predictions/cli_test_result.jpg",
        help="Path to save the annotated output image"
    )
    parser.add_argument(
        "--audio_output",
        type=str,
        default="outputs/sample_predictions/cli_test_alert.mp3",
        help="Path to save the generated spoken alert MP3 audio"
    )
    parser.add_argument(
        "--speak",
        action="store_true",
        help="Speak the warning aloud immediately through your computer speakers"
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found at: {image_path}")
        sys.exit(1)

    print("=" * 60)
    print("BLINDSPOT IMAGE & AUDIO TEST")
    print("=" * 60)
    print(f"Loading image: {image_path}")
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print("Error: Could not read image.")
        sys.exit(1)

    # Initialize system
    system = BlindSpotSystem()

    # Process frame
    print("Processing image through multimodal vision pipeline...")
    result = system.process_frame(img_bgr, include_walking_corridor=True)

    # 1. Print Obstacle Detections
    print("\n--- 1. Obstacle Detection & Spatial Zones ---")
    if result['obstacles']:
        for i, obj in enumerate(result['obstacles'], 1):
            print(f"  [{i}] {obj['class_name'].upper()} | Zone: {obj['zone'].upper()} | Confidence: {obj['confidence']*100:.1f}% | Centroid: {obj['center']}")
    else:
        print("  No hazardous obstacles detected in spatial zones.")

    # 2. Print Crosswalk Recognition
    print("\n--- 2. Zebra Crossing Recognition ---")
    cw = result['crosswalk']
    print(f"  Classification: {cw['predicted_class']} (Confidence: {cw['confidence']*100:.1f}%)")
    print(f"  Status: {cw['status_text']}")

    # 3. Print Spoken Audio Alert
    spoken_alert = result['fused_spoken_alert']
    priority = result['alert_priority']
    print(f"\n--- 3. Multimodal Spoken Warning (Priority: {priority}) ---")
    print(f"  >> \"{spoken_alert}\" <<")

    # 4. Save MP3 audio file
    if result['audio_bytes']:
        out_audio_path = Path(args.audio_output)
        out_audio_path.parent.mkdir(parents=True, exist_ok=True)
        out_audio_path.write_bytes(result['audio_bytes'])
        print(f"\n[Saved Audio] Spoken alert saved to: {out_audio_path.resolve()}")
    else:
        print("\n[Audio Notice] Audio MP3 generation fell back to text (network offline).")

    # 5. Save Annotated Output Image
    out_img_path = Path(args.output)
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img_path), result['composite_image'])
    print(f"[Saved Image] Annotated image saved to: {out_img_path.resolve()}")

    # 6. Optional direct speaker playback
    if args.speak:
        print("\nSpeaking alert aloud through system speakers...")
        spoken = speak_offline_pyttsx3(spoken_alert)
        if not spoken:
            print("System audio driver unavailable for direct speech.")

    print("\nTotal Latency: {} ms (~{} FPS on CPU)".format(
        result['latencies']['total_ms'], result['fps']
    ))
    print("=" * 60)

if __name__ == "__main__":
    main()
