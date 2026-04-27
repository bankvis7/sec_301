"""
Shared helper for resolving which model weights to load.

Priority:
  1. Fine-tuned weights in models/ (most recent run)
  2. Pre-trained yolov8n.pt fallback
"""

from pathlib import Path


def get_model_path(base_dir: str = ".") -> str:
    """
    Return the path to the best available model weights.
    Searches for the most recently modified best.pt under <base_dir>/models/,
    falling back to yolov8n.pt if no fine-tuned weights exist.
    """
    models_dir = Path(base_dir) / "models"
    candidates = sorted(
        models_dir.glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) if models_dir.exists() else []

    if candidates:
        print(f"[model] Using fine-tuned weights: {candidates[0]}")
        return str(candidates[0])

    fallback = str(Path(base_dir) / "yolov8n.pt")
    print(f"[model] No fine-tuned weights found, using pre-trained: {fallback}")
    return fallback
