"""
layers.py
=========
QuantKAN layer: aggregates per-edge quantum-inspired activations into a
Kolmogorov-Arnold representation, Section 3.3 and 3.5, Equations 3, 4,
11, 12.

Kolmogorov-Arnold representation (Eq. 3):
    F(x_1,...,x_n) = sum_{q=1}^{2n+1} Phi_q( sum_{p=1}^{n} phi_{q,p}(x_p) )

Standard KAN edge function (Eq. 4), for reference/reduction comparisons:
    phi_{q,p}(x) = sum_i c_i * B_i(x)                (fixed B-spline basis)

QuantKAN layer forward rule with residual spline term (Eq. 11):
    h_q = sum_{p=1}^{n} psi_{q,p}(x_p) + sum_i c_{q,i} * B_i(x_bar_q)

Stacked-layer composition (Eq. 12):
    h^(l) = QuantKANLayer_l( h^(l-1) ),  l = 1, ..., L

Implementation note
--------------------
Every edge (q, p) of a QuantKAN layer has its own instance of the
quantum-inspired activation unit psi_{q,p} (Eq. 5-9). For a layer with
n_in inputs and n_out outputs there are n_in * n_out such edges, each
with its own {alpha, beta, W, b, w} (Section 3.4). Rather than
instantiating n_in * n_out separate `nn.Module` objects (which is both
conceptually faithful and extremely slow in pure Python/PyTorch due to
per-module Python-loop overhead), this implementation stores all
per-edge parameters as single batched tensors of shape
(n_out, n_in, ...) and computes every edge's activation in parallel via
vectorized tensor ops. The two implementations are mathematically
identical; `activation.py`'s `QuantumInspiredEdgeActivation` is kept as
a small, easy-to-read single-edge reference module (useful for unit
tests and the Section 3.5.1-style numerical walkthrough), while this
module is the one actually used inside `QuantKANEV` for tractable
training speed at the widths reported in Table 5 (up to 64x64 = 4096
edges in a single layer).
"""

import math

import torch
import torch.nn as nn


def _bspline_basis(x: torch.Tensor, num_terms: int) -> torch.Tensor:
    """
    Lightweight fixed B-spline-like basis expansion used for the residual
    low-resolution term of Eq. 11 (sum_i c_{q,i} * B_i(x_bar_q)).

    A cosine basis over a bounded, tanh-squashed input is used as a
    smooth, numerically stable stand-in for a true B-spline basis; any
    proper B-spline implementation (e.g. `scipy.interpolate.BSpline`,
    fit once on a fixed grid of G knots per Table 8) can be substituted
    here without changing the surrounding layer/aggregation logic.

    Parameters
    ----------
    x : Tensor (..., n_in) -- the per-node "mean" input x_bar_q used for
        the residual path.
    num_terms : int -- number of basis functions (spline_basis_terms).

    Returns
    -------
    basis : Tensor (..., n_in, num_terms)
    """
    x_bounded = torch.tanh(x).unsqueeze(-1)                       # (..., n_in, 1)
    orders = torch.arange(1, num_terms + 1, device=x.device, dtype=x.dtype)
    basis = torch.cos(orders * x_bounded * math.pi / 2)           # (..., n_in, num_terms)
    return basis


class AdaptiveSplineFusionResidual(nn.Module):
    """
    Residual low-resolution spline path: sum_i c_{q,i} * B_i(x_bar_q),
    the second summand of Eq. 11. Kept trainable across all QuantKAN
    layers to stabilize training (per the manuscript's clarification
    that this residual term is applied at every stacked layer, with the
    dedicated "+1 fusion" stage at the Layer-3 transition re-applying
    the same mechanism once more before the prediction heads).
    """

    def __init__(self, n_in: int, n_out: int, num_basis_terms: int = 8):
        super().__init__()
        self.num_basis_terms = num_basis_terms
        # c_{q,i}: one coefficient vector per output node q, over n_in inputs
        # and num_basis_terms basis functions each.
        self.coeffs = nn.Parameter(0.01 * torch.randn(n_out, n_in, num_basis_terms))

    def forward(self, x_bar: torch.Tensor) -> torch.Tensor:
        """
        x_bar : Tensor (..., n_in)
        returns : Tensor (..., n_out)
        """
        basis = _bspline_basis(x_bar, self.num_basis_terms)         # (..., n_in, T)
        # sum over n_in and T: einsum with per-output coefficients c_{q,i}
        out = torch.einsum("...it,qit->...q", basis, self.coeffs)   # (..., n_out)
        return out


class QuantKANLayer(nn.Module):
    """
    One QuantKAN layer: for each output node q (of n_out) and each input
    node p (of n_in), a dedicated quantum-inspired edge activation
    psi_{q,p} (Eq. 5-8) is applied to x_p, then summed, plus the
    residual spline term (Eq. 11).

    All n_in * n_out edges' {alpha, beta, W, b, w} parameters (Section
    3.4) are stored as batched tensors and evaluated with vectorized
    tensor ops (see module docstring for rationale).

    Parameters
    ----------
    diagonal_coupling : bool
        If True, reproduces ablation variant (b), "w/o entanglement
        coupling W" (Table 19): the coupling matrix W is fixed to the
        identity (non-trainable) for every edge, so channels do not
        interact (Eq. 10's coupling-gap term is identically zero).
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        num_gates: int = 3,
        num_basis_terms: int = 8,
        diagonal_coupling: bool = False,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.Q = num_gates
        self.diagonal_coupling = diagonal_coupling

        # Per-edge rotation-gate parameters (Eq. 6): shape (n_out, n_in, Q).
        # Init: alpha_k <- 1, beta_k <- 0 for all edges (Algorithm 1, line 1).
        self.alpha = nn.Parameter(torch.ones(n_out, n_in, self.Q))
        self.beta = nn.Parameter(torch.zeros(n_out, n_in, self.Q))

        # Per-edge coupling matrix and bias (Eq. 7): shape (n_out, n_in, Q, Q) / (n_out, n_in, Q).
        eye = torch.eye(self.Q).expand(n_out, n_in, self.Q, self.Q).clone()
        if diagonal_coupling:
            # Ablation variant (b): W fixed to identity, not trainable.
            self.register_buffer("W", eye)
        else:
            self.W = nn.Parameter(eye.clone())
        self.b = nn.Parameter(torch.zeros(n_out, n_in, self.Q))

        # Per-edge output-mixing weights (Eq. 8): shape (n_out, n_in, Q).
        self.w = nn.Parameter(torch.ones(n_out, n_in, self.Q) / self.Q)

        self.residual = AdaptiveSplineFusionResidual(n_in, n_out, num_basis_terms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : Tensor (..., n_in)
        returns : Tensor (..., n_out)  -- h^(l), Eq. 12
        """
        # Stage 1 (Eq. 5): theta(x) = pi * tanh(x), broadcast over (n_out, n_in).
        theta = math.pi * torch.tanh(x)                              # (..., n_in)
        theta_exp = theta.unsqueeze(-2).unsqueeze(-1)                # (..., 1, n_in, 1)

        # Stage 2 (Eq. 6): R_k = cos(alpha_k * theta + beta_k) for every edge.
        gates = torch.cos(self.alpha * theta_exp + self.beta)        # (..., n_out, n_in, Q)

        # Stage 3 (Eq. 7): z(x) = W . gates + b, per edge.
        z = torch.einsum("...oiq,oiqr->...oir", gates, self.W) + self.b  # (..., n_out, n_in, Q)

        # Stage 4 (Eq. 8): psi_{q,p}(x_p) = sum_k w_k * tanh(z_k), per edge.
        psi_edges = torch.sum(self.w * torch.tanh(z), dim=-1)        # (..., n_out, n_in)

        # sum_{p=1}^{n} psi_{q,p}(x_p), Eq. 11's first summand.
        edge_term = psi_edges.sum(dim=-1)                             # (..., n_out)

        residual_term = self.residual(x)                              # (..., n_out), Eq. 11's 2nd term
        h_q = edge_term + residual_term                                # Eq. 11
        return h_q

    def total_edge_params(self) -> int:
        """(3Q + Q^2) trainable parameters per edge x n_in x n_out edges,
        per Section 3.4's per-edge parameter count (Eq. 5-8)."""
        per_edge = 3 * self.Q + (self.Q ** 2 if not self.diagonal_coupling else 0)
        return self.n_in * self.n_out * per_edge

    @torch.no_grad()
    def set_reduction_limit(self):
        """
        Configure every edge in this layer at the Eq. 9 reduction limit:
            W -> I, only gate 0 retained, w_1 -> 1, b -> 0
        Used to reproduce the "w/o rotation gates" ablation (Table 19,
        variant a), under which each edge is restored to a form
        structurally similar to the standard KAN edge function (Eq. 4).
        """
        if not self.diagonal_coupling:
            self.W.copy_(torch.eye(self.Q).expand_as(self.W))
        self.b.zero_()
        new_w = torch.zeros_like(self.w)
        new_w[..., 0] = 1.0
        self.w.copy_(new_w)


class InputProjection(nn.Module):
    """Layer 0 of Table 5: Linear + LayerNorm, d' -> hidden_dim."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x))
