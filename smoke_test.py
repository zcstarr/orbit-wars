#!/usr/bin/env python3
"""In-memory smoke test for the Orbit Wars conditional DDPM.

Exercises the full model + diffusion path (ObsEncoder -> cond -> ActionDenoiser
-> DDPM loss/backward/sample) on random tensors, with no parquet/cache I/O.
Catches shape, conditioning, masked-MSE, and reverse-process bugs in ~seconds
on CPU. Run: `python smoke_test.py` (exits non-zero on failure).
"""

from __future__ import annotations

import torch

from train_orbit import (
    ACTION_DIM,
    ACTION_HORIZON,
    GLOBAL_FEATURE_DIM,
    P_MAX,
    PLANET_FEATURE_DIM,
    FeatureConfig,
    ModelConfig,
    build_loss_mask,
    build_model,
    sample_actions,
)
from src.methods.ddpm import DDPM


def make_fake_batch(batch_size: int, device: torch.device) -> dict:
    """Random observation + target batch matching OrbitWarsDataset shapes."""
    g = torch.Generator(device="cpu").manual_seed(0)
    active = torch.rand(batch_size, P_MAX, generator=g) > 0.3
    # guarantee every row has >=1 active planet so masked pools aren't empty
    active[:, 0] = True
    return {
        "global_features": torch.randn(batch_size, GLOBAL_FEATURE_DIM, generator=g).to(device),
        "planet_features": torch.randn(batch_size, P_MAX, PLANET_FEATURE_DIM, generator=g).to(device),
        "planet_active_mask": active.to(device),
        "planet_source_mask": (torch.rand(batch_size, P_MAX, generator=g) > 0.6).to(device),
        "action_target": torch.randn(batch_size, ACTION_HORIZON, P_MAX, ACTION_DIM, generator=g).to(device),
        "action_source_mask": (torch.rand(batch_size, ACTION_HORIZON, P_MAX, generator=g) > 0.6).to(device),
        "time_valid_mask": torch.ones(batch_size, ACTION_HORIZON, dtype=torch.bool, device=device),
    }


def run_smoke(batch_size: int = 8, device: str = "cpu") -> None:
    dev = torch.device(device)
    feature_cfg = FeatureConfig()
    model_cfg = ModelConfig(num_timesteps=20)

    encoder, denoiser = build_model(feature_cfg, model_cfg)
    encoder = encoder.to(dev)
    denoiser = denoiser.to(dev)
    ddpm = DDPM(
        denoiser,
        dev,
        num_timesteps=model_cfg.num_timesteps,
        beta_start=model_cfg.beta_start,
        beta_end=model_cfg.beta_end,
    )

    batch = make_fake_batch(batch_size, dev)

    # --- training step: encode -> cond -> masked DDPM loss -> backward ---
    encoder.train()
    denoiser.train()
    cond = encoder(batch)
    assert cond["board_token"].shape == (batch_size, 1, model_cfg.d_model)
    assert cond["global_token"].shape == (batch_size, 1, model_cfg.d_model)

    loss, metrics = ddpm.compute_loss(
        batch["action_target"], cond=cond, loss_mask=build_loss_mask(batch)
    )
    assert loss.ndim == 0, "loss should be a scalar"
    assert torch.isfinite(loss), "loss is not finite"

    ddpm.zero_grad(set_to_none=True)
    encoder.zero_grad(set_to_none=True)
    loss.backward()

    den_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in denoiser.parameters())
    enc_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
    assert den_grad, "denoiser received no finite gradients"
    assert enc_grad, "encoder received no finite gradients (cond not wired into loss?)"

    # --- sampling step: encode -> conditional reverse diffusion ---
    actions = sample_actions(ddpm, encoder, batch)
    expected = (batch_size, ACTION_HORIZON, P_MAX, ACTION_DIM)
    assert tuple(actions.shape) == expected, f"sample shape {tuple(actions.shape)} != {expected}"
    assert torch.isfinite(actions).all(), "sampled actions contain non-finite values"

    n_launch = int((actions[:, 0, :, 0] > 0).sum())
    print(f"[ok] loss={metrics['mse']:.4f}  grads: enc={enc_grad} den={den_grad}")
    print(f"[ok] sample shape={tuple(actions.shape)}  turn-0 launches across batch={n_launch}")
    print("smoke test PASSED")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Orbit Wars DDPM in-memory smoke test")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    run_smoke(batch_size=args.batch_size, device=args.device)
