"""
Fine-tune YOLOv8n on COCO128 for person detection in crowded scenes.

COCO128 is a built-in ultralytics dataset (128 images, YOLO format) that
downloads automatically on first run — no manual setup required.

Usage:
    python train.py                        # default: COCO128, 20 epochs
    python train.py --epochs 50            # more epochs
    python train.py --dataset coco128      # explicit (same as default)
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


DATASET_YAMLS = {
    "coco128": "coco128.yaml",   # 128-image demo set, auto-downloads
    "coco":    "coco.yaml",      # full COCO 2017, auto-downloads (~20 GB)
}


def train(dataset: str, epochs: int, imgsz: int, batch: int, base_model: str) -> None:
    model = YOLO(base_model)

    results = model.train(
        model=base_model,
        data=DATASET_YAMLS[dataset],
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="crowd_counter",
        project="models",
        patience=10,
        save=True,
        device=0 if _has_gpu() else "cpu",
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\n[OK] Best weights saved to: {best_weights}")
    else:
        print(f"\n[WARN] best.pt not found in {results.save_dir}")
    return results


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for crowd counting")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_YAMLS.keys()),
        default="coco128",
        help="Dataset to train on (default: coco128, auto-downloads)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--base-model",
        default="yolov8n.pt",
        help="Starting weights (yolov8n.pt / yolov8s.pt / …)",
    )
    args = parser.parse_args()

    train(args.dataset, args.epochs, args.imgsz, args.batch, args.base_model)


if __name__ == "__main__":
    main()
