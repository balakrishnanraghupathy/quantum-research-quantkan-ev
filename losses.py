"""
losses.py
=========
Multi-task loss formulation, Section 3.6, Equations 13-16.

    L_SoH  = (1/B) sum_i (y_hat_i - y_i)^2                            (13)
    L_fade = (1/BH) sum_i sum_k (C_hat_{i,t+k} - C_{i,t+k})^2          (14)
    L_reg  = rho_1 * sum(alpha_k^2 + beta_k^2) + rho_2 * sum(c_{q,i}^2) (15)
    L_total = lambda_1*L_SoH + lambda_2*L_fade + lambda_3*L_reg        (16)
"""

from typing import Tuple

import torch
import torch.nn as nn

from .model import QuantKANEV


def soh_loss(soh_hat: torch.Tensor, soh_true: torch.Tensor) -> torch.Tensor:
    """Eq. 13: mean squared error over the batch."""
    return torch.mean((soh_hat - soh_true) ** 2)


def fade_loss(c_forecast: torch.Tensor, c_true: torch.Tensor) -> torch.Tensor:
    """
    Eq. 14: mean squared error over the batch AND the H-step horizon.

    c_forecast, c_true : Tensor (batch, H)
    """
    return torch.mean((c_forecast - c_true) ** 2)


def regularization_loss(model: QuantKANEV, rho1: float, rho2: float) -> torch.Tensor:
    """
    Eq. 15: penalizes rotation-gate parameters (alpha, beta) and residual
    spline coefficients (c_{q,i}) across every QuantKAN layer / edge in
    the model.
    """
    alpha_beta_sq = torch.tensor(0.0)
    spline_coeff_sq = torch.tensor(0.0)

    for module in model.modules():
        # Rotation-gate parameters live on QuantumInspiredEdgeActivation instances.
        if hasattr(module, "alpha") and hasattr(module, "beta"):
            alpha_beta_sq = alpha_beta_sq + torch.sum(module.alpha ** 2) + torch.sum(
                module.beta ** 2
            )
        # Residual spline coefficients live on AdaptiveSplineFusionResidual instances.
        if hasattr(module, "coeffs"):
            spline_coeff_sq = spline_coeff_sq + torch.sum(module.coeffs ** 2)

    return rho1 * alpha_beta_sq + rho2 * spline_coeff_sq


def total_loss(
    model: QuantKANEV,
    soh_hat: torch.Tensor,
    soh_true: torch.Tensor,
    c_forecast: torch.Tensor,
    c_true: torch.Tensor,
    loss_weights: Tuple[float, float, float],
    rho1: float,
    rho2: float,
) -> Tuple[torch.Tensor, dict]:
    """
    Eq. 16: L_total = lambda_1*L_SoH + lambda_2*L_fade + lambda_3*L_reg.

    Returns
    -------
    total : scalar Tensor
    components : dict of the individual (unweighted) loss terms, for logging.
    """
    lam1, lam2, lam3 = loss_weights

    l_soh = soh_loss(soh_hat, soh_true)
    l_fade = fade_loss(c_forecast, c_true)
    l_reg = regularization_loss(model, rho1, rho2)

    total = lam1 * l_soh + lam2 * l_fade + lam3 * l_reg
    components = {
        "L_SoH": l_soh.item(),
        "L_fade": l_fade.item(),
        "L_reg": l_reg.item(),
        "L_total": total.item(),
    }
    return total, components
