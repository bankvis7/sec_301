"""
Crowd-size estimator CLI
Usage:
    python crowd_counter.py --source <image|video|directory>
    python crowd_counter.py --source image.jpg --conf 0.4 --show
    python crowd_counter.py --source video.mp4 --save
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PERSON_CLASS = 0  # COCO class index for "person"


def count_frame(model, frame: np.ndarray, conf: float) -> tuple[int, np.ndarray]:
    """Return (count, annotated_frame) for a single BGR frame."""
    results = model(frame, conf=conf, classes=[PERSON_CLASS], verbose=False)
    r = results[0]
    annotated = r.plot()
    count = len(r.boxes)
    return count, annotated


def run(source: str, model_path: str, conf: float, show: bool, save: bool) -> None:
    model = YOLO(model_path)
    source_path = Path(source)

    if not source_path.exists():
        sys.exit(f"[ERROR] Source not found: {source}")

    # Image or video?
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    suffix = source_path.suffix.lower()

    if source_path.is_dir():
        images = sorted(p for p in source_path.iterdir() if p.suffix.lower() in image_exts)
        for img_path in images:
            _process_image(model, img_path, conf, show, save)
    elif suffix in image_exts:
        _process_image(model, source_path, conf, show, save)
    elif suffix in video_exts:
        _process_video(model, source_path, conf, show, save)
    else:
        sys.exit(f"[ERROR] Unsupported file type: {suffix}")


def _process_image(model, path: Path, conf: float, show: bool, save: bool) -> None:
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"[WARN] Cannot read {path}, skipping.")
        return
    count, annotated = count_frame(model, frame, conf)
    print(f"{path.name}: {count} people detected")
    if show:
        cv2.imshow("People Counter", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save:
        out_path = path.parent / f"{path.stem}_detected{path.suffix}"
        cv2.imwrite(str(out_path), annotated)
        print(f"  Saved to {out_path}")


def _process_video(model, path: Path, conf: float, show: bool, save: bool) -> None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {path}")

    writer = None
    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = path.parent / f"{path.stem}_detected.mp4"
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    frame_num = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count, annotated = count_frame(model, frame, conf)
            print(f"Frame {frame_num:04d}: {count} people")
            if show:
                cv2.imshow("People Counter", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if writer:
                writer.write(annotated)
            frame_num += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count people in images or videos"
    )
    parser.add_argument("--source", required=True, help="Image, video, or directory path")
    parser.add_argument(
        "--model", default="yolov8n.pt", help="YOLOv8 weights (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument("--show", action="store_true", help="Display output in a window")
    parser.add_argument("--save", action="store_true", help="Save annotated output to disk")
    args = parser.parse_args()

    run(args.source, args.model, args.conf, args.show, args.save)


if __name__ == "__main__":
    main()
