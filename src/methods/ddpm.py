"""
Denoising Diffusion Probabilistic Models (DDPM)
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class DDPM(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self.method_cfg = {
            "num_timesteps": self.num_timesteps,
            "beta_start": float(beta_start),
            "beta_end": float(beta_end),
        }
        self.register_buffer("betas", torch.linspace(
            beta_start, beta_end, num_timesteps, device=device))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(self.alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars",
                             torch.sqrt(1.0 - self.alpha_bars))

    def _extract(self, buffer: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Gather buffer[t] and reshape to broadcast against x of any rank."""
        out = buffer.to(device=t.device)[t]          # (B,)
        return out.view(t.shape[0], *([1] * (x.ndim - 1)))  # (B,1,...,1)

    def _eps(self, x_t: torch.Tensor, t: torch.Tensor, cond: Optional[Any]) -> torch.Tensor:
        """Call the noise predictor, with or without observation conditioning."""
        if cond is None:
            return self.model(x_t, t)
        return self.model(x_t, t, cond)

    # =========================================================================
    # Forward process
    # =========================================================================

    def forward_process(self, x_0, t):
        eps = torch.randn_like(x_0)
        sqrt_ab = self._extract(self.sqrt_alpha_bars, t, x_0)
        sqrt_1m_ab = self._extract(self.sqrt_one_minus_alpha_bars, t, x_0)
        x_t = sqrt_ab * x_0 + sqrt_1m_ab * eps
        return x_t, eps

    # =========================================================================
    # Training loss
    # =========================================================================

    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """MSE on predicted noise. Pass `cond` to condition the predictor and
        `loss_mask` (broadcastable to eps) to score only valid action slots."""
        cond = kwargs.get("cond")
        loss_mask = kwargs.get("loss_mask")
        b = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=self.device)
        x_t, eps = self.forward_process(x_0, t)
        eps_theta = self._eps(x_t, t, cond)
        loss = _masked_mse(eps_theta, eps, loss_mask)
        return loss, {'mse': loss.item()}

    # =========================================================================
    # Reverse process (sampling)
    # =========================================================================

    @torch.no_grad()
    def reverse_process(self, x_t: torch.Tensor, t: torch.Tensor, cond: Optional[Any] = None) -> torch.Tensor:
        eps_theta = self._eps(x_t, t, cond)
        sigma_t = torch.sqrt(self._extract(self.betas, t, x_t))
        z = torch.randn_like(x_t) if t[0] > 0 else torch.zeros_like(x_t)
        x_prev = 1/torch.sqrt(self._extract(self.alphas, t, x_t)) * (x_t - (
            (self._extract(self.betas, t, x_t)/self._extract(self.sqrt_one_minus_alpha_bars, t, x_t))) * eps_theta) + sigma_t * z
        return x_prev

    @torch.no_grad()
    def sample_trajectory(self, batch_size, image_shape, cond: Optional[Any] = None):
        # Full ancestral sampling: every t from T-1..0. Partial-step sampling
        # (e.g. DDIM) is intentionally not implemented; you cannot just lower the
        # step count with this schedule (it would index the wrong noise levels).
        self.eval_mode()
        x_t = torch.randn(batch_size, *image_shape, device=self.device)
        trajectory = [x_t.clone()]
        for i in range(self.num_timesteps):
            t = torch.full((batch_size,), self.num_timesteps - i - 1,
                           dtype=torch.long, device=self.device)
            x_t = self.reverse_process(x_t, t, cond)
            trajectory.append(x_t.clone())
        return trajectory

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, ...],
        **kwargs
    ) -> torch.Tensor:
        # Full ancestral sampling over the trained schedule (T steps). See
        # sample_trajectory: a reduced step count is not supported here.
        cond = kwargs.get('cond')
        self.eval_mode()
        x_t = torch.randn(batch_size, *image_shape, device=self.device)
        for i in range(self.num_timesteps):
            t = torch.full((batch_size,), self.num_timesteps - i - 1,
                           dtype=torch.long, device=self.device)
            x_t = self.reverse_process(x_t, t, cond)
        return x_t

    # =========================================================================
    # Device / state
    # =========================================================================

    def to(self, device: torch.device) -> "DDPM":
        super().to(device)
        self.device = device
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["method_cfg"] = dict(self.method_cfg)
        return state

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        super().load_state_dict(state_dict, strict=strict)
        if "method_cfg" in state_dict:
            self.method_cfg.update(state_dict["method_cfg"])

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "DDPM":
        ddpm_config = config.get("ddpm", config)
        return cls(
            model=model,
            device=device,
            num_timesteps=ddpm_config["num_timesteps"],
            beta_start=ddpm_config["beta_start"],
            beta_end=ddpm_config["beta_end"],
        )


def _masked_mse(pred: torch.Tensor, target: torch.Tensor,
                loss_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Plain MSE when loss_mask is None; otherwise mean over masked entries.

    loss_mask is broadcast up to pred's rank (e.g. [B,H,P] -> [B,H,P,1] over the
    action-feature dim), so only valid action slots contribute to the loss.
    """
    if loss_mask is None:
        return F.mse_loss(pred, target)
    mask = loss_mask.to(dtype=pred.dtype)
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    diff2 = (pred - target) ** 2
    denom = mask.sum() * pred.shape[-1]
    if denom <= 0:
        return diff2.mean()
    return (diff2 * mask).sum() / denom
