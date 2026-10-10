"""Shared exact-normalization and target-network helpers for the paper figures.

These definitions are copied from the manuscript repository at ac3c057.  They
are deliberately independent of the exploratory training-grid implementation.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


PARAMETERS = tuple(2**power for power in range(11))
WIDTHS = (4, 8, 16, 32)
TEACHERS = tuple(range(6))


def exact_ln(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=-1, keepdim=True)
    scale = torch.sqrt(torch.mean(centered.square(), dim=-1, keepdim=True))
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    normalized = centered / safe_scale
    return torch.where(scale > 0, normalized, torch.zeros_like(normalized))


def exact_rms(values: torch.Tensor) -> torch.Tensor:
    scale = torch.sqrt(torch.mean(values.square(), dim=-1, keepdim=True))
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    normalized = values / safe_scale
    return torch.where(scale > 0, normalized, torch.zeros_like(normalized))


def activate(values: torch.Tensor, name: str) -> torch.Tensor:
    if name == "relu":
        return torch.relu(values)
    if name == "leaky_relu":
        return torch.nn.functional.leaky_relu(values, negative_slope=0.01)
    if name == "absolute":
        return values.abs()
    if name == "silu":
        return torch.nn.functional.silu(values)
    if name == "tanh":
        return torch.tanh(values)
    if name == "sine":
        return torch.sin(values)
    if name == "x_sin_x":
        return values * torch.sin(values)
    if name == "exponential":
        return torch.exp(values)
    if name == "sinc":
        # torch.sinc(u) = sin(pi u)/(pi u), including the value 1 at u=0.
        return torch.sinc(values / math.pi)
    if name == "gaussian":
        return torch.exp(-values.square())
    raise ValueError(name)


class Teacher(nn.Module):
    def __init__(
        self,
        first_weight: np.ndarray,
        first_bias: np.ndarray,
        readout: np.ndarray,
        output_bias: float,
        activation_name: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.register_buffer("first_weight", torch.as_tensor(first_weight, dtype=dtype))
        self.register_buffer("first_bias", torch.as_tensor(first_bias, dtype=dtype))
        self.register_buffer("readout", torch.as_tensor(readout, dtype=dtype))
        self.register_buffer("output_bias", torch.tensor(output_bias, dtype=dtype))
        self.activation_name = activation_name

    def preactivation(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.first_weight.T + self.first_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            activate(self.preactivation(x), self.activation_name)
            @ self.readout[:, None]
            + self.output_bias
        )


def make_teacher_parameters(
    seed: int, width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    first_weight = 1.15 * rng.standard_normal((width, 1))
    first_bias = 0.45 * rng.standard_normal(width)
    readout = (0.70 * rng.standard_normal(width) + 0.25) / math.sqrt(width)
    output_bias = float(0.15 * rng.standard_normal())
    return first_weight, first_bias, readout, output_bias


def two_anchor_values(activation_name: str) -> tuple[float, float]:
    if activation_name == "relu":
        return 1.0, 0.0
    if activation_name == "silu":
        return float(1.2 / (1.0 + math.exp(-1.2))), float(
            -0.8 / (1.0 + math.exp(0.8))
        )
    if activation_name == "tanh":
        return math.tanh(1.2), math.tanh(-0.8)
    if activation_name == "sine":
        return 1.0, -1.0
    if activation_name == "sinc":
        return 1.0, 0.0
    if activation_name == "gaussian":
        return 1.0, math.exp(-1.0)
    raise ValueError(activation_name)
