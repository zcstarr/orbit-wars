"""
Base Class for Generative Methods

This module defines the abstract interface that generative methods
(e.g., DDPM) must implement. This ensures consistency across different
implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class BaseMethod(nn.Module, ABC):
    """
    Abstract base class for generative modeling methods.

    Methods (e.g., DDPM) should inherit from this class
    and implement the required methods.

    Attributes:
        model: The neural network used for prediction
        device: Device to run computations on
    """

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__()
        self.model = model
        self.device = device
        self.method_cfg: Dict[str, Any] = {}
        self.model_cfg: Dict[str, Any] = {}
        self.feature_cfg: Dict[str, Any] = {}

    @abstractmethod
    def compute_loss(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pass

    @abstractmethod
    def sample(
        self,
        batch_size: int,
        sample_shape: Tuple[int, ...],
        **kwargs,
    ) -> torch.Tensor:
        pass

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def to(self, device: torch.device) -> "BaseMethod":
        self.model = self.model.to(device)
        self.device = device
        return self

    def parameters(self):
        return self.model.parameters()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "method_cfg": dict(self.method_cfg),
            "model_cfg": dict(self.model_cfg),
            "feature_cfg": dict(self.feature_cfg),
        }

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        self.model.load_state_dict(state_dict["model"], strict=strict)
        self.method_cfg = dict(state_dict.get("method_cfg", {}))
        self.model_cfg = dict(state_dict.get("model_cfg", {}))
        self.feature_cfg = dict(state_dict.get("feature_cfg", {}))
