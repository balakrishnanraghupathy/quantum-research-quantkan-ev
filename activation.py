"""
activation.py
=============
The quantum-inspired learnable edge activation unit psi_{q,p}(.),
Section 3.4, Equations 5-9.

Every edge (p, q) of a QuantKAN layer is assigned one instance of this
four-stage unit. Given a scalar edge input x, the unit computes:

    Stage 1 (Eq. 5): theta(x)      = pi * tanh(x)
    Stage 2 (Eq. 6): R_y,R_z,R_x   = cos(alpha_k * theta(x) + beta_k),  k = 1..Q
    Stage 3 (Eq. 7): z(x)          = W . [R_y, R_z, R_x]^T + b
    Stage 4 (Eq. 8): psi(x)        = sum_k  w_k * tanh(z_k(x))

All parameters of one unit {alpha, beta, W, b, w} are trained jointly
with the rest of the network (Section 3.6).

Terminology note (see manuscript Section 3.4 clarification): the labels
R_y, R_z, R_x denote three independently-parameterized gate channels
inspired by single-qubit Pauli-rotation generators. Each channel shares
the identical functional form cos(alpha_k*theta+beta_k) and is
distinguished only by its own trainable (alpha_k, beta_k); this is a
classically-executable simplification, not a literal composition of
physical rotation operators.

Reduction check (Eq. 9): as W -> I, Q -> 1, w_1 -> 1, b -> 0, this unit
collapses to psi(x) -> tanh(cos(alpha_1*theta(x) + beta_1)), which is
structurally comparable to a single B-spline segment of the standard
KAN edge function (Eq. 4). See `reduction_limit_state_dict()` below for
a helper that configures a unit at exactly this limiting case, used to
reproduce the "w/o rotation gates" ablation (Table 19, variant a).
"""

import math
import torch
import torch.nn as nn


class QuantumInspiredEdgeActivation(nn.Module):
    """
    psi_{q,p}(x): learnable per-edge activation unit (Eq. 5-8).

    Parameters
    ----------
    num_gates : int
        Number of rotation-gate channels Q (default 3, per Table 8).
    """

    def __init__(self, num_gates: int = 3):
        super().__init__()
        self.Q = num_gates

        # Stage 2: per-gate scale (alpha_k) and offset (beta_k), init alpha=1, beta=0
        # (Algorithm 1, line 1: "Initialize Theta (Xavier); set alpha_k <- 1, beta_k <- 0
        #  for all edges")
        self.alpha = nn.Parameter(torch.ones(self.Q))
        self.beta = nn.Parameter(torch.zeros(self.Q))

        # Stage 3: entanglement-inspired coupling matrix W (Q x Q) and bias b (Q,)
        # Initialized to identity + zero bias, consistent with Algorithm 1's stated
        # initialization (W = I, b = 0) for the numerical-illustration walkthrough
        # (Section 3.5.1); subsequent training moves W away from I as needed.
        self.W = nn.Parameter(torch.eye(self.Q))
        self.b = nn.Parameter(torch.zeros(self.Q))

        # Stage 4: output mixing weights w_k
        self.w = nn.Parameter(torch.ones(self.Q) / self.Q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (...,) -- scalar edge input per element.

        Returns
        -------
        psi : Tensor of shape (...,) -- scalar activation output per element.
        """
        # Stage 1 (Eq. 5): theta(x) = pi * tanh(x)
        theta = math.pi * torch.tanh(x)                      # (...,)

        # Stage 2 (Eq. 6): R_k = cos(alpha_k * theta + beta_k) for k = 1..Q
        theta_exp = theta.unsqueeze(-1)                       # (..., 1)
        gates = torch.cos(self.alpha * theta_exp + self.beta)  # (..., Q)

        # Stage 3 (Eq. 7): z(x) = W . gates^T + b
        z = torch.matmul(gates, self.W.T) + self.b            # (..., Q)

        # Stage 4 (Eq. 8): psi(x) = sum_k w_k * tanh(z_k(x))
        psi = torch.sum(self.w * torch.tanh(z), dim=-1)       # (...,)
        return psi

    def num_trainable_params(self) -> int:
        """3Q + Q^2 trainable parameters per edge (Section 3.4, Eq. 8 discussion)."""
        return 3 * self.Q + self.Q ** 2

    @torch.no_grad()
    def set_reduction_limit(self):
        """
        Configure this unit at the Eq. 9 reduction limit:
            W -> I, Q -> 1 (only gate 0 retained), w_1 -> 1, b -> 0
        Used to reproduce the "w/o rotation gates" ablation (Table 19,
        variant a), under which each edge is restored to a form
        structurally similar to the standard KAN edge function (Eq. 4).
        Only gate index 0 is left active; gates 1..Q-1 are masked to a
        no-op by setting their output weight to zero.
        """
        self.W.copy_(torch.eye(self.Q))
        self.b.zero_()
        new_w = torch.zeros(self.Q)
        new_w[0] = 1.0
        self.w.copy_(new_w)


class DiagonalCouplingEdgeActivation(QuantumInspiredEdgeActivation):
    """
    Ablation variant (b): "w/o entanglement coupling W" (Table 19).

    The trainable coupling matrix W of Eq. 7 is replaced by a fixed
    identity matrix (i.e. no inter-channel coupling / diagonal-only
    interaction), and the coupling-gap term C_{q,p}(x) of Eq. 10 is
    therefore identically zero. All other stages (Eq. 5, 6, 8) are
    unchanged and remain trainable.
    """

    def __init__(self, num_gates: int = 3):
        super().__init__(num_gates)
        self.W.requires_grad_(False)
        with torch.no_grad():
            self.W.copy_(torch.eye(self.Q))
