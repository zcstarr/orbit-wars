"""Reconstruct a loss-vs-step curve from saved checkpoints.

The trainer never persisted loss (only stdout/wandb), but every checkpoint
stores its `step` + full model. So we re-evaluate each `step_*.pt` over a FIXED
set of batches with a FIXED RNG seed (so the sampled diffusion timestep `t` and
forward noise are identical across checkpoints) and plot masked-MSE vs step.

Usage:
    python plot_loss.py                       # all checkpoints, valid split
    python plot_loss.py --stride 20           # every 20th checkpoint (faster)
    python plot_loss.py --split train --n-batches 8
"""

import argparse
import json
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

import train_orbit as T

REPO_ROOT = pathlib.Path(__file__).resolve().parent
STEP_RE = re.compile(r"step_(\d+)\.pt$")


def find_checkpoints(ckpt_dir: pathlib.Path, stride: int):
    ckpts = []
    for p in ckpt_dir.glob("step_*.pt"):
        m = STEP_RE.search(p.name)
        if m:
            ckpts.append((int(m.group(1)), p))
    ckpts.sort(key=lambda x: x[0])
    return ckpts[::stride] if stride > 1 else ckpts


@torch.no_grad()
def eval_loss(ddpm, encoder, batches, seed: int) -> float:
    encoder.eval()
    ddpm.eval_mode()
    # Same seed per checkpoint => identical t + noise => comparable curve.
    torch.manual_seed(seed)
    total, n = 0.0, 0
    for batch in batches:
        x_0 = batch["action_target"]
        cond = encoder(batch)
        loss_mask = T.build_loss_mask(batch)
        loss, _ = ddpm.compute_loss(x_0, cond=cond, loss_mask=loss_mask)
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description="Plot loss vs step from checkpoints")
    ap.add_argument("--checkpoint-dir", type=pathlib.Path, default=REPO_ROOT / "checkpoints")
    ap.add_argument("--cache-dir", type=pathlib.Path, default=REPO_ROOT / "cache")
    ap.add_argument("--split", choices=["valid", "train", "test"], default="valid")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth checkpoint")
    ap.add_argument("--n-batches", type=int, default=4, help="batches per checkpoint")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=REPO_ROOT / "loss_curve.png")
    args = ap.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ckpts = find_checkpoints(args.checkpoint_dir, args.stride)
    if not ckpts:
        raise SystemExit(f"no step_*.pt found in {args.checkpoint_dir}")
    print(f"evaluating {len(ckpts)} checkpoints on '{args.split}' split, device={device}")

    # Build loaders once; prefetch a fixed batch set onto the device so every
    # checkpoint sees the exact same data.
    cfg = T.TrainConfig(cache_dir=args.cache_dir)
    cfg.batch_size = args.batch_size
    cfg.num_workers = 0
    cfg.device = str(device)
    manifest = pd.read_parquet(args.cache_dir / "samples.parquet")
    train_loader, valid_loader, test_loader = T.make_loaders(manifest, cfg)
    loader = {"train": train_loader, "valid": valid_loader, "test": test_loader}[args.split]

    batches = []
    for i, batch in enumerate(loader):
        if i >= args.n_batches:
            break
        batches.append(T._move_batch(batch, device))

    rows = []
    for step, path in ckpts:
        ddpm, encoder, _, _ = T.load_checkpoint(path, device)
        mse = eval_loss(ddpm, encoder, batches, args.seed)
        rows.append({"step": step, "mse": mse})
        print(f"step={step:>9d}  mse={mse:.5f}")

    df = pd.DataFrame(rows)
    csv_path = args.out.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df["mse"], lw=1)
    plt.xlabel("step")
    plt.ylabel(f"masked MSE ({args.split}, {args.n_batches} fixed batches)")
    plt.title("Recomputed loss vs step")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"\nwrote {args.out} and {csv_path}")
    print(json.dumps({"min_mse": float(df['mse'].min()),
                      "min_at_step": int(df.loc[df['mse'].idxmin(), 'step']),
                      "last_mse": float(df['mse'].iloc[-1])}, indent=2))


if __name__ == "__main__":
    main()
