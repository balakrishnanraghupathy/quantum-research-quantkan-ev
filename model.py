"""
model.py
========
Full QuantKAN-EV backbone with dual-head prediction module, assembled
per Table 5's layer-wise configuration:

    Layer 0    Input Projection      d' -> 64      Linear + LayerNorm
    Layer 1-2  QuantKAN Layers 1-2   64 -> 64 -> 48 psi(.) per edge, Q gates
    Layer 3    Adaptive Spline Fusion 48 -> 32      residual low-res path (Eq. 11)
    Layer 4a   SoH Head              32 -> 1        sigmoid-scaled linear output
    Layer 4b   Capacity-Fade Head    32 -> H        linear projection (Eq. 17-18)

The SoH and capacity-fade heads share the same 32-dimensional latent
representation h^(L) (Section 3.5, "dual-head prediction module"),
which is the source of the ~49% parameter saving over two independent
single-task networks reported in Table 9.
"""

from typing import Tuple

import torch
import torch.nn as nn

from .layers import InputProjection, QuantKANLayer
from .capacity_fade import CapacityFadeHead
from .config import QuantKANEVConfig


class SoHHead(nn.Module):
    """
    Layer 4a of Table 5: quantum-inspired unit + sigmoid-scaled linear
    output, producing a single scalar SoH estimate in [0, 1].
    """

    def __init__(self, latent_dim: int = 32, num_gates: int = 3):
        super().__init__()
        self.refine = QuantKANLayer(n_in=latent_dim, n_out=latent_dim, num_gates=num_gates)
        self.out = nn.Linear(latent_dim, 1)

    def forward(self, h_L: torch.Tensor) -> torch.Tensor:
        refined = self.refine(h_L)
        soh = torch.sigmoid(self.out(refined)).squeeze(-1)   # (...,) in [0, 1]
        return soh


class QuantKANEV(nn.Module):
    """
    Full QuantKAN-EV model: Input Projection -> QuantKAN Layer 1 ->
    QuantKAN Layer 2 -> Adaptive Spline Fusion -> {SoH head, Capacity-Fade
    head}.

    Forward pass expects a windowed input tensor of shape
    (batch, window_length, d') and internally pools over the window
    (mean pooling over time) before the backbone, consistent with the
    windowed-tensor description of Stage II (Section 3.2): "the output
    is a w x d' Tensor per sample."
    """

    def __init__(self, config: QuantKANEVConfig):
        super().__init__()
        self.config = config

        self.input_proj = InputProjection(config.input_dim, config.hidden_dim)

        self.layer1 = QuantKANLayer(
            n_in=config.hidden_dim,
            n_out=config.hidden_dim,
            num_gates=config.num_rotation_gates,
            num_basis_terms=config.spline_basis_terms,
        )
        self.layer2 = QuantKANLayer(
            n_in=config.hidden_dim,
            n_out=config.layer2_dim,
            num_gates=config.num_rotation_gates,
            num_basis_terms=config.spline_basis_terms,
        )
        self.fusion = QuantKANLayer(
            n_in=config.layer2_dim,
            n_out=config.fusion_out_dim,
            num_gates=config.num_rotation_gates,
            num_basis_terms=config.spline_basis_terms,
        )

        self.soh_head = SoHHead(config.fusion_out_dim, config.num_rotation_gates)
        self.fade_head = CapacityFadeHead(
            latent_dim=config.fusion_out_dim,
            horizon=config.forecast_horizon,
            num_gates=config.num_rotation_gates,
        )

    def backbone(self, x_window: torch.Tensor) -> torch.Tensor:
        """
        x_window : Tensor (batch, window_length, d')
        returns  : Tensor (batch, fusion_out_dim)  -- shared latent h^(L)
        """
        x_pooled = x_window.mean(dim=1)               # (batch, d') - window pooling
        h0 = self.input_proj(x_pooled)                 # Layer 0
        h1 = self.layer1(h0)                            # Layer 1 (Eq. 12, l=1)
        h2 = self.layer2(h1)                             # Layer 2 (Eq. 12, l=2)
        h_L = self.fusion(h2)                             # Layer 3 (Adaptive Spline Fusion)
        return h_L

    def forward(
        self, x_window: torch.Tensor, c_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x_window : Tensor (batch, window_length, d') -- windowed input tensor.
        c_t : Tensor (batch,) -- current capacity estimate, used as the
              anchor for the monotonic capacity-fade forecast (Eq. 18).

        Returns
        -------
        soh_hat : Tensor (batch,)          -- SoH estimate in [0, 1].
        c_forecast : Tensor (batch, H)     -- non-increasing capacity-fade trajectory.
        """
        h_L = self.backbone(x_window)
        soh_hat = self.soh_head(h_L)
        c_forecast = self.fade_head(h_L, c_t)
        return soh_hat, c_forecast

    def count_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)
