"""
Denoising Diffusion Probabilistic Models (DDPM)
"""

from src.methods.base import BaseMethod
from torch.distributions import Normal
import math
from typing import Dict, Tuple, Optional, Literal, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class DDPM2(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
        # TODO: Add your own arguments here
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        # TODO: Implement your own init
        self.register_buffer("betas", torch.linspace(
            beta_start, beta_end, num_timesteps, device=device))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(self.alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars",
                             torch.sqrt(1.0 - self.alpha_bars))

    # =========================================================================
    # You can add, delete or modify as many functions as you would like
    # =========================================================================

    # Pro tips: If you have a lot of pseudo parameters that you will specify for each
    # model run but will be fixed once you specified them (say in your config),
    # then you can use super().register_buffer(...) for these parameters

    # Pro tips 2: If you need a specific broadcasting for your tensors,
    # it's a good idea to write a general helper function for that

    # =========================================================================
    # Forward process
    # =========================================================================

    def forward_process(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # TODO: Add your own arguments here
        # TODO: Implement the forward (noise adding) process of DDPM
        # calculate Normal
        eps = Normal(0, 1).sample()
        eps = torch.randn_like(x_0)
        x_t_given_x0 = torch.sqrt(
            self.alpha_bars[t].view(-1, 1)) * x_0 + self.sqrt_one_minus_alpha_bars[t].view(-1, 1) * eps
        return x_t_given_x0, eps

    # =========================================================================
    # Training loss
    # =========================================================================

    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        eps = kwargs.get('eps', None)
        eps_theta = kwargs.get('eps_theta', None)
        if eps is None or eps_theta is None:
            raise ValueError("eps and eps_theta is required")

        loss = F.mse_loss(eps_theta, eps)
        return loss, {'mse': loss.item()}
        """
        TODO: Implement your DDPM loss function here

        Args:
            x_0: Clean data samples of shape (batch_size, channels, height, width)
            **kwargs: Additional method-specific arguments
        
        Returns:
            loss: Scalar loss tensor for backpropagation
            metrics: Dictionary of metrics for logging (e.g., {'mse': 0.1})
        """

        return

    # =========================================================================
    # Reverse process (sampling)
    # =========================================================================

    @torch.no_grad()
    def reverse_process(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        TODO: Implement one step of the DDPM reverse process

        Args:
            x_t: Noisy samples at time t (batch_size, channels, height, width)
            t: the time
            **kwargs: Additional method-specific arguments

        Returns:
            x_prev: Noisy samples at time t-1 (batch_size, channels, height, width)
        """
        eps_theta = self.model(x_t, t)
        sigma_t = torch.sqrt(self.betas[t]).view(-1, 1)
        z = torch.randn_like(sigma_t)
        x_prev = 1/torch.sqrt(self.alphas[t].view(-1, 1)) * (x_t - (
            (self.betas[t].view(-1, 1)/self.sqrt_one_minus_alpha_bars[t].view(-1, 1))) * eps_theta) + sigma_t * z
        return x_prev

    @torch.no_grad()
    def sample_trajectory(self, batch_size, image_shape, num_steps):
        self.eval_mode()
        x_t = torch.randn(batch_size, *image_shape, device=self.device)
        trajectory = [x_t.clone()]
        for i in range(num_steps):
            t = torch.full((batch_size,), num_steps - i - 1,
                           dtype=torch.long, device=self.device)
            x_t = self.reverse_process(x_t, t)
            trajectory.append(x_t.clone())
        return trajectory

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        # TODO: add your arguments here
        **kwargs
    ) -> torch.Tensor:
        num_steps = kwargs['num_steps']
        """
        TODO: Implement DDPM sampling loop: start from pure noise, iterate through all the time steps using reverse_process()

        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            **kwargs: Additional method-specific arguments (e.g., num_steps)
        
        Returns:
            samples: Generated samples of shape (batch_size, *image_shape)
        """
        self.eval_mode()
        x_t = torch.randn(batch_size, *image_shape,
                          device=self.device)  # (100, 1)
        for i in range(num_steps):
            t = torch.full((batch_size,), num_steps - i - 1,
                           dtype=torch.long, device=self.device)
            x_t = self.reverse_process(x_t, t)
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
        state["num_timesteps"] = self.num_timesteps
        # TODO: add other things you want to save
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "DDPM":
        ddpm_config = config.get("ddpm", config)
        return cls(
            model=model,
            device=device,
            num_timesteps=ddpm_config["num_timesteps"],
            beta_start=ddpm_config["beta_start"],
            beta_end=ddpm_config["beta_end"],
            # TODO: add your parameters here
        )


class DDPM2(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
        # TODO: Add your own arguments here
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self. beta_start = beta_start
        self. beta_end = beta_end
        # TODO: Implement your own init

    # =========================================================================
    # You can add, delete or modify as many functions as you would like
    # =========================================================================

    # Pro tips: If you have a lot of pseudo parameters that you will specify for each
    # model run but will be fixed once you specified them (say in your config),
    # then you can use super().register_buffer(...) for these parameters

    # Pro tips 2: If you need a specific broadcasting for your tensors,
    # it's a good idea to write a general helper function for that

    # =========================================================================
    # Forward process
    # =========================================================================

    def forward_process(self):  # TODO: Add your own arguments here
        # TODO: Implement the forward (noise adding) process of DDPM
        raise NotImplementedError

    # =========================================================================
    # Training loss
    # =========================================================================

    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        TODO: Implement your DDPM loss function here

        Args:
            x_0: Clean data samples of shape (batch_size, channels, height, width)
            **kwargs: Additional method-specific arguments

        Returns:
            loss: Scalar loss tensor for backpropagation
            metrics: Dictionary of metrics for logging (e.g., {'mse': 0.1})
        """

        raise NotImplementedError

    # =========================================================================
    # Reverse process (sampling)
    # =========================================================================

    @torch.no_grad()
    def reverse_process(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        TODO: Implement one step of the DDPM reverse process

        Args:
            x_t: Noisy samples at time t (batch_size, channels, height, width)
            t: the time
            **kwargs: Additional method-specific arguments

        Returns:
            x_prev: Noisy samples at time t-1 (batch_size, channels, height, width)
        """
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        # TODO: add your arguments here
        **kwargs
    ) -> torch.Tensor:
        """
        TODO: Implement DDPM sampling loop: start from pure noise, iterate through all the time steps using reverse_process()

        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            **kwargs: Additional method-specific arguments (e.g., num_steps)

        Returns:
            samples: Generated samples of shape (batch_size, *image_shape)
        """
        self.eval_mode()
        raise NotImplementedError

    # =========================================================================
    # Device / state
    # =========================================================================

    def to(self, device: torch.device) -> "DDPM":
        super().to(device)
        self.device = device
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["num_timesteps"] = self.num_timesteps
        # TODO: add other things you want to save
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "DDPM":
        ddpm_config = config.get("ddpm", config)
        return cls(
            model=model,
            device=device,
            num_timesteps=ddpm_config["num_timesteps"],
            beta_start=ddpm_config["beta_start"],
            beta_end=ddpm_config["beta_end"],
            # TODO: add your parameters here
        )


class DDPM(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
        # TODO: Add your own arguments here
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        # TODO: Implement your own init
        self.register_buffer("betas", torch.linspace(
            beta_start, beta_end, num_timesteps, device=device))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(self.alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars",
                             torch.sqrt(1.0 - self.alpha_bars))

    # =========================================================================
    # You can add, delete or modify as many functions as you would like
    # =========================================================================

    # Pro tips: If you have a lot of pseudo parameters that you will specify for each
    # model run but will be fixed once you specified them (say in your config),
    # then you can use super().register_buffer(...) for these parameters

    # Pro tips 2: If you need a specific broadcasting for your tensors,
    # it's a good idea to write a general helper function for that
    def _extract(self, buffer: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Gather buffer[t] and reshape to broadcast against x of any rank."""
        out = buffer.to(device=t.device)[t]          # (B,)
        return out.view(t.shape[0], *([1] * (x.ndim - 1)))  # (B,1,...,1)

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
        b = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=self.device)
        x_t, eps = self.forward_process(x_0, t)
        eps_theta = self.model(x_t, t)
        loss = F.mse_loss(eps_theta, eps)
        return loss, {'mse': loss.item()}
        """
        TODO: Implement your DDPM loss function here

        Args:
            x_0: Clean data samples of shape (batch_size, channels, height, width)
            **kwargs: Additional method-specific arguments
        
        Returns:
            loss: Scalar loss tensor for backpropagation
            metrics: Dictionary of metrics for logging (e.g., {'mse': 0.1})

        return
        """

    # =========================================================================
    # Reverse process (sampling)
    # =========================================================================

    @torch.no_grad()
    def reverse_process(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        TODO: Implement one step of the DDPM reverse process

        Args:
            x_t: Noisy samples at time t (batch_size, channels, height, width)
            t: the time
            **kwargs: Additional method-specific arguments

        Returns:
            x_prev: Noisy samples at time t-1 (batch_size, channels, height, width)
        """

        eps_theta = self.model(x_t, t)
        sigma_t = torch.sqrt(self._extract(self.betas, t, x_t))
        z = torch.randn_like(x_t) if t[0] > 0 else torch.zeros_like(x_t)
        x_prev = 1/torch.sqrt(self._extract(self.alphas, t, x_t)) * (x_t - (
            (self._extract(self.betas, t, x_t)/self._extract(self.sqrt_one_minus_alpha_bars, t, x_t))) * eps_theta) + sigma_t * z
        return x_prev

    @torch.no_grad()
    def sample_trajectory(self, batch_size, image_shape, num_steps):
        self.eval_mode()
        x_t = torch.randn(batch_size, *image_shape, device=self.device)
        trajectory = [x_t.clone()]
        for i in range(num_steps):
            t = torch.full((batch_size,), num_steps - i - 1,
                           dtype=torch.long, device=self.device)
            x_t = self.reverse_process(x_t, t)
            trajectory.append(x_t.clone())
        return trajectory

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        # TODO: add your arguments here
        **kwargs
    ) -> torch.Tensor:
        num_steps = kwargs.get('num_steps', self.num_timesteps)
        """
        TODO: Implement DDPM sampling loop: start from pure noise, iterate through all the time steps using reverse_process()

        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            **kwargs: Additional method-specific arguments (e.g., num_steps)
        
        Returns:
            samples: Generated samples of shape (batch_size, *image_shape)
        """
        self.eval_mode()
        x_t = torch.randn(batch_size, *image_shape,
                          device=self.device)  # (100, 1)
        for i in range(num_steps):
            t = torch.full((batch_size,), num_steps - i - 1,
                           dtype=torch.long, device=self.device)
            x_t = self.reverse_process(x_t, t)
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
        state["num_timesteps"] = self.num_timesteps
        # TODO: add other things you want to save
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "DDPM":
        ddpm_config = config.get("ddpm", config)
        return cls(
            model=model,
            device=device,
            num_timesteps=ddpm_config["num_timesteps"],
            beta_start=ddpm_config["beta_start"],
            beta_end=ddpm_config["beta_end"],
            # TODO: add your parameters here
        )
