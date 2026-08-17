"""
domain_adapt.py
================
Cross-domain validation strategy, Section 3.8, Equation 19, Algorithm 2.

Feature-statistics alignment (Eq. 19):
    x' = ((x - mu_t) / sigma_t) * sigma_s + mu_s

where mu_s, sigma_s are the pooled source-domain statistics and
mu_t, sigma_t are the target-domain (EV) statistics; this re-expresses
target features on the source domain's scale before they reach the
frozen backbone layers.

Selective freezing / fine-tuning rule (Algorithm 2, authoritative
specification -- see manuscript's clarification note in Section 3.8):
    - FROZEN:      rotation-angle parameters (alpha, beta) of QuantKAN
                   layers 1 through L-1 (the general degradation
                   representation learned during source pretraining).
    - FINE-TUNED:  coupling (W) and output-mixing (w, b) weights of the
                   FINAL layer (the Adaptive Spline Fusion layer) and
                   the prediction heads (SoH head, capacity-fade head),
                   at a reduced learning rate eta_a = eta / 10.

This corresponds to approximately 18% of total backbone parameters
being frozen (Table 9), leaving the final layer and heads free to
specialize to the field-data characteristics of the target domain.
"""

from typing import List

import torch
import torch.nn as nn

from .model import QuantKANEV
from .activation import QuantumInspiredEdgeActivation


def compute_domain_statistics(x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Computes per-channel mean and standard deviation of a domain's
    feature tensor, used as (mu, sigma) in Eq. 19.

    x : Tensor (N, d') -- pooled feature vectors from one domain.
    Returns (mu, sigma), each of shape (d',).
    """
    mu = x.mean(dim=0)
    sigma = x.std(dim=0) + 1e-8  # numerical stability
    return mu, sigma


def align_features(
    x_target: torch.Tensor,
    mu_source: torch.Tensor,
    sigma_source: torch.Tensor,
    mu_target: torch.Tensor,
    sigma_target: torch.Tensor,
) -> torch.Tensor:
    """Eq. 19: statistics-only feature alignment of target -> source scale."""
    return ((x_target - mu_target) / sigma_target) * sigma_source + mu_source


def apply_selective_freezing(model: QuantKANEV) -> List[nn.Parameter]:
    """
    Freezes the rotation-angle parameters (alpha, beta) of QuantKAN
    layers 1 through L-1 (i.e. `model.layer1` and `model.layer2`, the
    early/general-representation layers per Table 5), and leaves the
    final Adaptive Spline Fusion layer (`model.fusion`) and both
    prediction heads (`model.soh_head`, `model.fade_head`) trainable in
    full, per Algorithm 2's authoritative specification.

    Returns the list of parameters that remain trainable, for use when
    constructing the fine-tuning optimizer (see train_adapt.py).
    """
    # Step 1: freeze everything by default.
    for p in model.parameters():
        p.requires_grad_(False)

    # Step 2: re-enable gradients on the final layer + both heads (full
    # fine-tuning of these, including their coupling/output-mixing weights).
    for module in [model.fusion, model.soh_head, model.fade_head]:
        for p in module.parameters():
            p.requires_grad_(True)

    # Step 3: the input projection is also considered part of "layers 1
    # through L-1" and stays frozen, consistent with preserving the
    # general degradation representation learned on the source domain.
    # (model.input_proj, model.layer1, model.layer2 remain frozen.)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return trainable_params


def frozen_fraction(model: QuantKANEV) -> float:
    """
    Reports the fraction of total backbone parameters that are frozen
    under `apply_selective_freezing`, for comparison against the ~18%
    figure reported in Table 9.
    """
    total = sum(p.numel() for p in model.parameters())
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return frozen / total if total > 0 else 0.0
