"""Single-image infrared small-target detection / inference.

Runs one (or two) trained ACM checkpoints on a single real-world image and
produces a comparison figure:  input | saliency heatmap | binary mask | overlay
with detected target bounding boxes (via connected components).

Typical use -- compare baseline (paper ACM) vs improved (ACM++):
  python src/infer.py --image my_shot.png \
      --ckpt-before runs/acm_fpn/best.pth \
      --ckpt-after  runs/acm_fpn_pp/best.pth \
      --out results/compare_my_shot.png

Single model:
  python src/infer.py --image my_shot.png --ckpt runs/acm_fpn/best.pth
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models import build_model

# Must match dataset.py normalization.
_MEAN, _STD = 0.485, 0.229


def load_checkpoint(net, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck.get("args", {})
    model = build_model(
        net, in_channels=1,
        block_per_stage=a.get("block", 3),
        fusion=a.get("fusion", "acm"),
        downsample=a.get("downsample", "adjusted"),
        spatial_gate=a.get("spatial_gate", False),
        deep_supervision=a.get("deep_supervision", False),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def preprocess(image_path, base_size):
    """Load -> grayscale -> resize to base_size. Returns (tensor, display_gray).

    display_gray is the [0,1] grayscale used for visualization (pre-norm).
    """
    img = Image.open(image_path).convert("L")
    orig_size = img.size  # (W, H)
    if base_size:
        img = img.resize((base_size, base_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    norm = (arr - _MEAN) / _STD
    tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)  # 1x1xHxW
    return tensor, arr, orig_size


@torch.no_grad()
def predict(model, tensor, device, thresh):
    logits = model(tensor.to(device))
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (prob > thresh).astype(np.uint8)
    return prob, mask


def detect_boxes(mask, min_area=2):
    """Connected-component bounding boxes from a binary mask.

    Uses OpenCV if available, else a simple numpy flood fill. Returns list of
    (x, y, w, h).
    """
    boxes = []
    try:
        import cv2
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area >= min_area:
                boxes.append((int(x), int(y), int(w), int(h)))
    except Exception:
        # fallback: bounding box of all foreground pixels grouped naively
        ys, xs = np.where(mask > 0)
        if len(xs):
            boxes.append((int(xs.min()), int(ys.min()),
                          int(xs.max() - xs.min() + 1),
                          int(ys.max() - ys.min() + 1)))
    return boxes


def _draw_panel(ax, base_gray, overlay=None, boxes=None, title="", cmap="gray"):
    ax.imshow(base_gray, cmap="gray")
    if overlay is not None:
        ax.imshow(overlay, cmap="jet", alpha=0.45)
    if boxes:
        for (x, y, w, h) in boxes:
            # pad small boxes so single-pixel targets stay visible
            pad = 4
            rect = patches.Rectangle((x - pad, y - pad), w + 2 * pad, h + 2 * pad,
                                     linewidth=1.5, edgecolor="lime",
                                     facecolor="none")
            ax.add_patch(rect)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def render_single(gray, prob, mask, boxes, out_path, title):
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    _draw_panel(ax[0], gray, title="input (grayscale)")
    _draw_panel(ax[1], gray, overlay=prob, title="saliency heatmap")
    ax[2].imshow(mask, cmap="gray"); ax[2].set_title("binary mask"); ax[2].axis("off")
    _draw_panel(ax[3], gray, boxes=boxes,
                title=f"detections: {len(boxes)} target(s)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_compare(gray, before, after, out_path):
    """before/after are dicts with keys prob, mask, boxes, label."""
    fig, ax = plt.subplots(2, 4, figsize=(14, 7))
    for row, res in enumerate([before, after]):
        _draw_panel(ax[row, 0], gray,
                    title=f"{res['label']}\ninput" if row == 0 else res['label'])
        _draw_panel(ax[row, 1], gray, overlay=res["prob"], title="saliency heatmap")
        ax[row, 2].imshow(res["mask"], cmap="gray")
        ax[row, 2].set_title("binary mask"); ax[row, 2].axis("off")
        _draw_panel(ax[row, 3], gray, boxes=res["boxes"],
                    title=f"detections: {len(res['boxes'])} target(s)")
    fig.suptitle("ACM single-image detection: before vs after improvement",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to a real-world image")
    ap.add_argument("--net", default="acm_fpn", choices=["acm_fpn", "acm_unet"])
    ap.add_argument("--ckpt", default=None, help="single-model checkpoint")
    ap.add_argument("--ckpt-before", default=None, help="baseline checkpoint")
    ap.add_argument("--ckpt-after", default=None, help="improved checkpoint")
    ap.add_argument("--base-size", type=int, default=480,
                    help="resize square size; set 0 to keep original size")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=2,
                    help="min connected-component area to count as a target")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_size = args.base_size if args.base_size > 0 else None
    tensor, gray, orig = preprocess(args.image, base_size)
    print(f"image={args.image} orig_size(WxH)={orig} infer_size={gray.shape}")

    stem = os.path.splitext(os.path.basename(args.image))[0]
    os.makedirs("results", exist_ok=True)

    if args.ckpt_before and args.ckpt_after:
        results = {}
        for key, ckpt, label in [
            ("before", args.ckpt_before, "BEFORE (paper ACM)"),
            ("after", args.ckpt_after, "AFTER (ACM++)"),
        ]:
            model = load_checkpoint(args.net, ckpt, device)
            prob, mask = predict(model, tensor, device, args.thresh)
            boxes = detect_boxes(mask, args.min_area)
            results[key] = {"prob": prob, "mask": mask, "boxes": boxes,
                            "label": label}
            print(f"  {label}: {len(boxes)} target(s), "
                  f"max prob={prob.max():.3f}")
        out = args.out or os.path.join("results", f"compare_{stem}.png")
        render_compare(gray, results["before"], results["after"], out)
        print(f"saved comparison -> {out}")
    else:
        ckpt = args.ckpt or args.ckpt_before or args.ckpt_after
        if not ckpt:
            raise SystemExit("provide --ckpt, or both --ckpt-before/--ckpt-after")
        model = load_checkpoint(args.net, ckpt, device)
        prob, mask = predict(model, tensor, device, args.thresh)
        boxes = detect_boxes(mask, args.min_area)
        print(f"  {len(boxes)} target(s), max prob={prob.max():.3f}")
        out = args.out or os.path.join("results", f"detect_{stem}.png")
        render_single(gray, prob, mask, boxes, out, title=f"ACM detection: {stem}")
        print(f"saved detection -> {out}")


if __name__ == "__main__":
    main()
