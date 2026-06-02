"""One-click reproduction driver.

Runs the full experiment suite by repeatedly invoking train.py and
evaluate.py as subprocesses, so each model trains in a clean process.

Presets:
  main      : ACM-FPN + ACM-U-Net (paper main result, Table 3 rows).
  ablation  : modulation-scheme ablation on FPN host (Table 2 left block):
              topdown_local / bilocal / biglobal / acm.
  downsample: down-sampling ablation (Table 2): adjusted vs regular, FPN host.
  full      : main + ablation + downsample.

After each run, evaluate.py is called on the test split and metrics are
collected into results/summary.json.

Usage (from project root, in the `acm` env):
  python src/run_all.py --preset main      --epochs 300
  python src/run_all.py --preset ablation  --epochs 300
  python src/run_all.py --preset full      --epochs 300
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable  # current interpreter (the acm env when launched from it)


def run_dir_tag(net, fusion="acm", downsample="adjusted", block=3):
    tag = net
    if fusion != "acm":
        tag += f"_{fusion}"
    if downsample != "adjusted":
        tag += f"_{downsample}"
    if block != 3:
        tag += f"_b{block}"
    return tag


def train_one(net, fusion, downsample, block, args):
    cmd = [PY, os.path.join(HERE, "train.py"),
           "--net", net, "--fusion", fusion, "--downsample", downsample,
           "--block", str(block),
           "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
           "--lr", str(args.lr), "--base-size", str(args.base_size),
           "--data-root", args.data_root]
    print("\n>>> TRAIN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def eval_one(net, fusion, downsample, block, args):
    tag = run_dir_tag(net, fusion, downsample, block)
    ckpt = os.path.join(ROOT, "runs", tag, "best.pth")
    metrics_path = os.path.join(ROOT, "results", tag, "test_metrics.json")
    cmd = [PY, os.path.join(HERE, "evaluate.py"),
           "--net", net, "--ckpt", ckpt, "--split", "test",
           "--base-size", str(args.base_size),
           "--data-root", args.data_root, "--block", str(block),
           "--vis", str(args.vis),
           "--vis-dir", os.path.join(ROOT, "results", tag, "vis"),
           "--metrics-out", metrics_path]
    print("\n>>> EVAL:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        m["tag"] = tag
        return m
    return {"tag": tag, "error": "metrics file missing"}


def jobs_for_preset(preset):
    """Return list of (net, fusion, downsample, block) tuples."""
    main = [("acm_fpn", "acm", "adjusted", 3),
            ("acm_unet", "acm", "adjusted", 3)]
    ablation = [("acm_fpn", "topdown_local", "adjusted", 3),
                ("acm_fpn", "bilocal", "adjusted", 3),
                ("acm_fpn", "biglobal", "adjusted", 3)]
    # ACM-FPN/adjusted is already in `main`, reused as the ablation's acm row.
    downsample = [("acm_fpn", "acm", "regular", 3)]
    if preset == "main":
        return main
    if preset == "ablation":
        return main[:1] + ablation       # FPN acm + 3 variants
    if preset == "downsample":
        return main[:1] + downsample      # adjusted (from main) + regular
    if preset == "full":
        # de-dup while preserving order
        seen, out = set(), []
        for j in main + ablation + downsample:
            if j not in seen:
                seen.add(j); out.append(j)
        return out
    raise ValueError(preset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="main",
                    choices=["main", "ablation", "downsample", "full"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--base-size", type=int, default=480)
    ap.add_argument("--data-root", default="data/sirst")
    ap.add_argument("--vis", type=int, default=12)
    ap.add_argument("--skip-train", action="store_true",
                    help="only evaluate existing checkpoints")
    args = ap.parse_args()

    jobs = jobs_for_preset(args.preset)
    print(f"preset={args.preset}: {len(jobs)} model(s) to run")
    for j in jobs:
        print("   ", j)

    summary = []
    for net, fusion, downsample, block in jobs:
        if not args.skip_train:
            train_one(net, fusion, downsample, block, args)
        m = eval_one(net, fusion, downsample, block, args)
        summary.append(m)
        print(f"   -> {m.get('tag')}: IoU={m.get('IoU')} nIoU={m.get('nIoU')}")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.path.join(ROOT, "results", f"summary_{args.preset}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n==== SUMMARY ({args.preset}) ====")
    for m in summary:
        print(f"  {m.get('tag'):28s} IoU={m.get('IoU'):.4f} "
              f"nIoU={m.get('nIoU'):.4f}" if "IoU" in m else f"  {m}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
