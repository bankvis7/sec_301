"""
Adversarial Evasion Attack against the crowd counting system (YOLOv8 person detector).

Attack type: White-box adversarial patch (FGSM / PGD variant)
Goal:        Suppress person-class confidence scores → cause undercounting.
Reference:   Thys et al. (2019) "Fooling automated surveillance cameras:
             adversarial patches to attack person detection", CVPR Workshops.

Usage:
    # Generate adversarial perturbation on a single image
    python attack.py --source image.jpg --output outputs/attacked.jpg

    # Evaluate detection count before and after attack
    python attack.py --source image.jpg --evaluate
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO


PERSON_CLASS = 0


def preprocess(frame: np.ndarray, imgsz: int = 640) -> torch.Tensor:
    """BGR ndarray → normalised CHW tensor [1, 3, H, W]."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (imgsz, imgsz))
    t = torch.from_numpy(resized).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0)


def postprocess_count(model: YOLO, frame: np.ndarray, conf: float = 0.5) -> int:
    """Run standard YOLOv8 inference and return person count."""
    results = model(frame, conf=conf, classes=[PERSON_CLASS], verbose=False)
    return len(results[0].boxes)


def fgsm_attack(
    model: YOLO,
    frame: np.ndarray,
    eps: float = 0.05,
    imgsz: int = 640,
) -> np.ndarray:
    """
    Fast Gradient Sign Method (FGSM) applied to the full image.
    Minimises the mean person-class score across all anchor predictions.

    Returns the adversarial frame as a BGR ndarray (same size as input).
    """
    nn_model = model.model
    nn_model.eval()
    device = next(nn_model.parameters()).device

    img_tensor = preprocess(frame, imgsz).to(device).requires_grad_(True)

    with torch.enable_grad():
        raw = nn_model(img_tensor)  # [1, 84, 8400] for COCO-trained model

        # YOLOv8 output: tuple where raw[0] has shape [1, 84, 8400].
        # First 4 channels = box (xywh); channels 4..83 = class probs.
        # Index 4 (channel 4) = person class (COCO class 0).
        person_scores = raw[0][0, 4, :]  # shape [8400]
        loss = torch.sigmoid(person_scores).mean()
        loss.backward()

    perturbation = eps * img_tensor.grad.data.sign()
    adv_tensor = (img_tensor - perturbation).clamp(0.0, 1.0)

    # Convert back to BGR ndarray at original resolution
    adv_np = adv_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    adv_bgr = cv2.cvtColor((adv_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.resize(adv_bgr, (frame.shape[1], frame.shape[0]))


def pgd_attack(
    model: YOLO,
    frame: np.ndarray,
    eps: float = 0.05,
    alpha: float = 0.005,
    iterations: int = 40,
    imgsz: int = 640,
) -> np.ndarray:
    """
    Projected Gradient Descent (PGD) attack — stronger multi-step variant.
    Iteratively applies FGSM steps while keeping the perturbation within
    the L-infinity ball of radius eps.
    """
    nn_model = model.model
    nn_model.eval()
    device = next(nn_model.parameters()).device

    original = preprocess(frame, imgsz).to(device)
    adv = original.clone().requires_grad_(True)

    for _ in range(iterations):
        if adv.grad is not None:
            adv.grad.zero_()

        with torch.enable_grad():
            raw = nn_model(adv)
            person_scores = raw[0][0, 4, :]  # [1, 84, 8400] → [8400]
            loss = torch.sigmoid(person_scores).mean()
            loss.backward()

        with torch.no_grad():
            adv_data = adv.data - alpha * adv.grad.sign()
            # Project back into eps-ball around original
            delta = torch.clamp(adv_data - original, -eps, eps)
            adv.data = torch.clamp(original + delta, 0.0, 1.0)

        adv.requires_grad_(True)

    adv_np = adv.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    adv_bgr = cv2.cvtColor((adv_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.resize(adv_bgr, (frame.shape[1], frame.shape[0]))


def evaluate(model: YOLO, original: np.ndarray, adversarial: np.ndarray, conf: float = 0.5) -> None:
    before = postprocess_count(model, original, conf)
    after = postprocess_count(model, adversarial, conf)
    reduction = before - after
    pct = (reduction / before * 100) if before > 0 else 0.0
    print(f"  People detected (clean)      : {before}")
    print(f"  People detected (adversarial): {after}")
    print(f"  Reduction                    : {reduction} ({pct:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial evasion attack")
    parser.add_argument("--source", required=True, help="Input image path")
    parser.add_argument("--output", default="outputs/attacked.jpg", help="Output path")
    parser.add_argument(
        "--attack",
        choices=["fgsm", "pgd"],
        default="pgd",
        help="Attack algorithm (default: pgd)",
    )
    parser.add_argument("--eps", type=float, default=0.05, help="Max L-inf perturbation (0..1)")
    parser.add_argument("--model", default="yolov8n.pt", help="Model weights")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--evaluate", action="store_true", help="Print count before/after")
    args = parser.parse_args()

    frame = cv2.imread(args.source)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {args.source}")

    model = YOLO(args.model)

    print(f"[*] Running {args.attack.upper()} attack (eps={args.eps}) …")
    if args.attack == "fgsm":
        adv_frame = fgsm_attack(model, frame, eps=args.eps)
    else:
        adv_frame = pgd_attack(model, frame, eps=args.eps)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, adv_frame)
    print(f"[+] Adversarial image saved to: {args.output}")

    if args.evaluate:
        print("\n[Evaluation]")
        evaluate(model, frame, adv_frame, args.conf)


if __name__ == "__main__":
    main()
