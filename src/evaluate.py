"""Evaluation + visualization for a trained ACM model on the SIRST test set."""

import argparse
import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SIRSTDataset
from metrics import SegMetrics
from models import build_model


def save_visualization(img, mask, pred, name, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = img.squeeze().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    mask = mask.squeeze().cpu().numpy()
    pred = pred.squeeze().cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("input")
    ax[1].imshow(mask, cmap="gray"); ax[1].set_title("ground truth")
    ax[2].imshow(pred, cmap="gray"); ax[2].set_title("prediction")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="acm_fpn", choices=["acm_fpn", "acm_unet"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="data/sirst")
    ap.add_argument("--base-size", type=int, default=480)
    ap.add_argument("--split", default="test")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--block", type=int, default=3)
    ap.add_argument("--vis", type=int, default=12, help="num samples to render")
    ap.add_argument("--vis-dir", default=None)
    ap.add_argument("--metrics-out", default=None,
                    help="explicit path for the metrics json (default keyed by net)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cargs = ckpt.get("args", {})
    block = cargs.get("block", args.block)
    fusion = cargs.get("fusion", "acm")
    downsample = cargs.get("downsample", "adjusted")
    spatial_gate = cargs.get("spatial_gate", False)
    deep_supervision = cargs.get("deep_supervision", False)

    model = build_model(args.net, in_channels=1, block_per_stage=block,
                        fusion=fusion, downsample=downsample,
                        spatial_gate=spatial_gate,
                        deep_supervision=deep_supervision).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = SIRSTDataset(args.data_root, args.split, args.base_size, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    vis_dir = args.vis_dir or os.path.join("results", args.net, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    metrics = SegMetrics(thresh=args.thresh)
    saved = 0
    with torch.no_grad():
        for img, mask, name in loader:
            img, mask = img.to(device), mask.to(device)
            logits = model(img)
            metrics.update(logits, mask)
            if saved < args.vis:
                pred = (torch.sigmoid(logits) > args.thresh).float()
                save_visualization(img[0], mask[0], pred[0], name[0], vis_dir)
                saved += 1

    res = metrics.compute()
    res["split"] = args.split
    res["net"] = args.net
    res["fusion"] = fusion
    res["downsample"] = downsample
    res["block"] = block
    res["n"] = len(ds)
    print(json.dumps(res, indent=2))
    out = args.metrics_out or os.path.join(
        "results", args.net, f"{args.split}_metrics.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved metrics -> {out}; visualizations -> {vis_dir}")


if __name__ == "__main__":
    main()
