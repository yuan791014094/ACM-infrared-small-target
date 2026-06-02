"""Training script for ACM-FPN / ACM-U-Net on the SIRST dataset.

Paper config: Soft-IoU loss, Nesterov SGD, lr=0.05, batch=8, 300 epochs,
He initialization. CPU training here -> epochs/batch are configurable for
practical reproduction; defaults follow the paper.
"""

import argparse
import os
import json
import time

import torch
from torch.utils.data import DataLoader

from dataset import SIRSTDataset
from metrics import SoftIoULoss, HybridLoss, SegMetrics
from models import build_model


def he_init(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                          nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    metrics = SegMetrics(thresh=0.5)
    total_loss, nb = 0.0, 0
    for img, mask, _ in loader:
        img, mask = img.to(device), mask.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(img)
            loss = criterion(logits, mask)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * img.size(0)
        nb += img.size(0)
        main_logits = logits[0] if isinstance(logits, (list, tuple)) else logits
        metrics.update(main_logits.detach(), mask)
    stats = metrics.compute()
    stats["loss"] = total_loss / max(nb, 1)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="acm_fpn",
                    choices=["acm_fpn", "acm_unet"])
    ap.add_argument("--fusion", default="acm",
                    choices=["acm", "topdown_global", "topdown_local",
                             "bilocal", "biglobal"],
                    help="cross-layer modulation scheme (Table 2 ablation)")
    ap.add_argument("--downsample", default="adjusted",
                    choices=["adjusted", "regular"],
                    help="down-sampling scheme (Table 2 ablation)")
    ap.add_argument("--data-root", default="data/sirst")
    ap.add_argument("--base-size", type=int, default=480)
    ap.add_argument("--block", type=int, default=3, help="b in paper Table 1")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None,
                    help="path to a *_last.pth checkpoint to resume from; "
                         "use 'auto' to resume from <out_dir>/last.pth if present")
    # ---- ACM++ improvement switches (paper-faithful baseline = all off) ----
    ap.add_argument("--improved", action="store_true",
                    help="enable all ACM++ improvements at once "
                         "(spatial gate + deep supervision + hybrid loss)")
    ap.add_argument("--spatial-gate", action="store_true",
                    help="ACM++ #1: CBAM spatial gate in fusion")
    ap.add_argument("--deep-supervision", action="store_true",
                    help="ACM++ #2: auxiliary head on coarser decoder level")
    ap.add_argument("--hybrid-loss", action="store_true",
                    help="ACM++ #3: SoftIoU + Focal hybrid loss")
    ap.add_argument("--focal-weight", type=float, default=1.0)
    args = ap.parse_args()

    # `--improved` is shorthand for turning on all three improvements.
    if args.improved:
        args.spatial_gate = True
        args.deep_supervision = True
        args.hybrid_loss = True

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.net
    if args.fusion != "acm":
        tag += f"_{args.fusion}"
    if args.downsample != "adjusted":
        tag += f"_{args.downsample}"
    if args.block != 3:
        tag += f"_b{args.block}"
    if args.spatial_gate or args.deep_supervision or args.hybrid_loss:
        tag += "_pp"  # ACM++ (improved)
    out_dir = args.out or os.path.join("runs", tag)
    os.makedirs(out_dir, exist_ok=True)

    train_set = SIRSTDataset(args.data_root, "train", args.base_size, augment=True)
    val_set = SIRSTDataset(args.data_root, "val", args.base_size, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=args.workers)

    model = build_model(args.net, in_channels=1, block_per_stage=args.block,
                        fusion=args.fusion, downsample=args.downsample,
                        spatial_gate=args.spatial_gate,
                        deep_supervision=args.deep_supervision).to(device)
    he_init(model)
    nparam = sum(p.numel() for p in model.parameters())

    if args.hybrid_loss:
        criterion = HybridLoss(focal_weight=args.focal_weight)
    else:
        criterion = SoftIoULoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, nesterov=True,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # ---- optional resume ----
    start_epoch = 1
    best_niou = -1.0
    history = []
    resume_path = args.resume
    if resume_path == "auto":
        cand = os.path.join(out_dir, "last.pth")
        resume_path = cand if os.path.exists(cand) else None
    if resume_path and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        if "scheduler" in ck:
            scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck.get("epoch", 0) + 1
        best_niou = ck.get("best_niou", -1.0)
        hist_path = os.path.join(out_dir, "history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)
        print(f"resumed from {resume_path} at epoch {start_epoch}")

    print(f"[{tag}] net={args.net} fusion={args.fusion} "
          f"downsample={args.downsample} improved={args.spatial_gate or args.deep_supervision or args.hybrid_loss} "
          f"params={nparam:,} device={device} "
          f"train={len(train_set)} val={len(val_set)} epochs={args.epochs}")

    for ep in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va = run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step()
        dt = time.time() - t0
        rec = {"epoch": ep, "lr": optimizer.param_groups[0]["lr"],
               "train_loss": tr["loss"], "train_IoU": tr["IoU"],
               "val_loss": va["loss"], "val_IoU": va["IoU"],
               "val_nIoU": va["nIoU"], "time": dt}
        history.append(rec)
        print(f"ep {ep:3d}/{args.epochs} | tr_loss {tr['loss']:.4f} "
              f"tr_IoU {tr['IoU']:.4f} | val_IoU {va['IoU']:.4f} "
              f"val_nIoU {va['nIoU']:.4f} | {dt:.1f}s", flush=True)

        if va["nIoU"] > best_niou:
            best_niou = va["nIoU"]
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "epoch": ep, "val": va},
                       os.path.join(out_dir, "best.pth"))
        # full state every epoch -> enables --resume
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "args": vars(args), "epoch": ep, "best_niou": best_niou},
                   os.path.join(out_dir, "last.pth"))
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"done. best val nIoU={best_niou:.4f}  -> {out_dir}/best.pth")


if __name__ == "__main__":
    main()
