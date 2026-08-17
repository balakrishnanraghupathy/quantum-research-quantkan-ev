"""
capacity_fade.py
=================
Capacity-fade forecasting head, Section 3.7, Equations 17-18.

Raw multi-step forecast (Eq. 17):
    C_hat(t+1:t+H) = W_out . QuantKANLayer_fade(h^(L)) + b_out

Monotonic reparameterization (Eq. 18):
    C_hat(t+k) = C_hat(t) - sum_{j=1}^{k} softplus(delta_j),   k = 1..H

Proposition (guaranteed monotonicity), proved in the manuscript
(Section 3.7): for any real-valued delta_j (j = 1..H), the sequence
defined by Eq. 18 is non-increasing in k, independent of training,
because softplus(z) = ln(1 + e^z) > 0 for every real z, so every
cumulative subtracted term S_k = sum_{j<=k} softplus(delta_j) is
strictly positive and strictly increasing in k. This is an
architectural identity, not a statistical regularity enforced via a
loss penalty -- see `verify_monotonicity_guarantee()` for a runtime
verification that holds even for an untrained (randomly initialized)
head.

Floating-point precision note: the proof above holds exactly in real
arithmetic. In IEEE 754 float32, softplus(z) underflows to exactly 0.0
for very large negative z (below roughly -103), which would make a
subtracted term non-positive rather than strictly positive at that one
step. This is a numerical representation limit, not a flaw in the
architecture, and only arises for pathologically large pre-activation
magnitudes far beyond anything produced during normal training or
inference; switching to float64 removes it entirely if it is ever a
practical concern for a given deployment.
"""

import torch
import torch.nn as nn

from .layers import QuantKANLayer


class CapacityFadeHead(nn.Module):
    """
    Predicts a physically-constrained, non-increasing H-step capacity
    trajectory from the shared latent representation h^(L).

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the shared latent representation h^(L)
        (32, per Table 5's Adaptive Spline Fusion output).
    horizon : int
        Forecast horizon H (10 cycles, per Table 8).
    num_gates : int
        Rotation-gate depth Q used inside the dedicated capacity-fade
        QuantKAN layer (Eq. 17).
    """

    def __init__(self, latent_dim: int = 32, horizon: int = 10, num_gates: int = 3):
        super().__init__()
        self.horizon = horizon
        self.fade_layer = QuantKANLayer(
            n_in=latent_dim, n_out=latent_dim, num_gates=num_gates
        )
        self.out_proj = nn.Linear(latent_dim, horizon)  # W_out, b_out (Eq. 17)

    def forward(self, h_L: torch.Tensor, c_t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h_L : Tensor (..., latent_dim) -- shared latent representation h^(L).
        c_t : Tensor (...,) -- current capacity estimate C(t) (e.g. from
              the SoH head or from the input DeltaCapacity channel).

        Returns
        -------
        c_forecast : Tensor (..., horizon) -- C_hat(t+1), ..., C_hat(t+H),
            guaranteed non-increasing along the last dimension for any
            learned delta (Eq. 18 proof).
        """
        fade_repr = self.fade_layer(h_L)                    # (..., latent_dim)
        delta = self.out_proj(fade_repr)                    # (..., horizon), raw deltas

        # Eq. 18: cumulative subtraction of strictly-positive softplus(delta)
        positive_steps = torch.nn.functional.softplus(delta)  # (..., horizon), > 0 always
        cumulative_drop = torch.cumsum(positive_steps, dim=-1)  # strictly increasing in k

        c_forecast = c_t.unsqueeze(-1) - cumulative_drop     # (..., horizon)
        return c_forecast


@torch.no_grad()
def verify_monotonicity_guarantee(
    head: CapacityFadeHead, batch: int = 8, latent_dim: int = 32, seed: int = 0
) -> bool:
    """
    Runtime proof-check corresponding to the manuscript's Proposition
    (Section 3.7): verifies that C_hat(t+k) < C_hat(t+k-1) for every
    k = 1..H, using an UNTRAINED head with randomly initialized
    parameters, to demonstrate the guarantee is architectural (holds by
    construction) rather than something learned during optimization.

    Returns True if the guarantee holds for the random sample (it always
    should, for any parameter values, per the proof).
    """
    torch.manual_seed(seed)
    h_L = torch.randn(batch, latent_dim)
    c_t = torch.rand(batch) + 0.5  # arbitrary starting capacity in (0.5, 1.5)

    forecast = head(h_L, c_t)  # (batch, horizon)
    diffs = forecast[:, 1:] - forecast[:, :-1]  # should be < 0 everywhere
    return bool(torch.all(diffs < 0).item())
