"""
Defence Wrapper
Implements input-transformation defences against adversarial evasion attacks.

Supported defences (can be stacked):
  jpeg      – JPEG re-compression (quality 75) removes high-frequency perturbations
  blur      – Gaussian blur (kernel 5×5) smooths pixel-level noise
  squeeze   – Feature squeezing: reduces bit depth to 4-bit to destroy fine perturbations
  median    – Median filter (3×3) non-linearly removes adversarial noise

Usage:
    python defend.py --source image.jpg --defence jpeg
    python defend.py --source outputs/attacked.jpg --defence jpeg blur --evaluate
    python defend.py --source outputs/attacked.jpg --defence all --evaluate
"""

import argparse
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

PERSON_CLASS = 0


# ── individual defences ──────────────────────────────────────────────────────

def jpeg_compression(frame: np.ndarray, quality: int = 75) -> np.ndarray:
    """Re-encode the image as JPEG to strip high-frequency adversarial noise."""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    result = np.array(Image.open(buf))
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def gaussian_blur(frame: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Gaussian blur smooths pixel-level perturbations."""
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)


def feature_squeezing(frame: np.ndarray, bit_depth: int = 4) -> np.ndarray:
    """
    Reduce colour bit-depth to destroy fine-grained adversarial perturbations.
    E.g. 4-bit squeezing quantises each channel into 16 levels.
    """
    levels = 2 ** bit_depth
    squeezed = np.round(frame.astype(np.float32) / 255.0 * (levels - 1))
    squeezed = (squeezed / (levels - 1) * 255).astype(np.uint8)
    return squeezed


def median_filter(frame: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Median filter — non-linear, effective against salt-and-pepper noise."""
    return cv2.medianBlur(frame, ksize)


DEFENCE_FN = {
    "jpeg": jpeg_compression,
    "blur": gaussian_blur,
    "squeeze": feature_squeezing,
    "median": median_filter,
}

# ── pipeline ─────────────────────────────────────────────────────────────────

def apply_defences(frame: np.ndarray, defences: list[str]) -> np.ndarray:
    result = frame.copy()
    for d in defences:
        result = DEFENCE_FN[d](result)
    return result


def count_people(model: YOLO, frame: np.ndarray, conf: float) -> int:
    results = model(frame, conf=conf, classes=[PERSON_CLASS], verbose=False)
    return len(results[0].boxes)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    all_defences = list(DEFENCE_FN.keys())

    parser = argparse.ArgumentParser(description="Defence wrapper")
    parser.add_argument("--source", required=True, help="Input image path")
    parser.add_argument("--output", default="outputs/defended.jpg", help="Output path")
    parser.add_argument(
        "--defence",
        nargs="+",
        choices=all_defences + ["all"],
        default=["jpeg"],
        help="Defence(s) to apply, or 'all' to apply all",
    )
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Compare person count on clean, attacked, and defended images",
    )
    parser.add_argument(
        "--attacked-source",
        default=None,
        help="Path to adversarial image for evaluation comparison",
    )
    args = parser.parse_args()

    defences = all_defences if "all" in args.defence else args.defence

    frame = cv2.imread(args.source)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {args.source}")

    defended = apply_defences(frame, defences)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, defended)
    print(f"[+] Defended image saved to: {args.output}")
    print(f"    Defences applied: {', '.join(defences)}")

    if args.evaluate:
        model = YOLO(args.model)
        clean_count = count_people(model, frame, args.conf)
        defended_count = count_people(model, defended, args.conf)

        print("\n[Evaluation]")
        print(f"  Clean image count   : {clean_count}")

        if args.attacked_source:
            attacked = cv2.imread(args.attacked_source)
            if attacked is not None:
                attacked_count = count_people(model, attacked, args.conf)
                print(f"  Attacked image count: {attacked_count}")

        print(f"  Defended image count: {defended_count}")
        recovery = defended_count - (count_people(model, cv2.imread(args.attacked_source), args.conf) if args.attacked_source else 0)
        print(f"  Recovery            : +{recovery} detections restored")


if __name__ == "__main__":
    main()
