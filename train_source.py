"""
train_source.py
================
Algorithm 1 -- Source-Domain Training of the Proposed QuantKAN-EV
Framework (Section 3.6).

    Input: Pooled source dataset D_s = D_s^NASA U D_s^CALCE; epochs E;
           learning rate eta; batch size B; layers L; grid size G; gates Q
    Output: Trained parameters Theta

    1: Initialize Theta (Xavier); set alpha_k<-1, beta_k<-0 for all edges
    2: Preprocess and pool D_s^NASA and D_s^CALCE
       (clean -> impute -> normalize -> window -> merge)
    3: Split D_s into training / validation folds (85% / 15%)
    4: for epoch = 1 to E do
    5:   for each mini-batch (Xb, yb) do
    6:     for layer l = 1 to L: compute psi_pq via Eq. 5-8, aggregate via Eq. 11
    7:     Compute outputs y_hat_b^SoH, C_hat_b; L_total via Eq. 13-16
    8:     Theta <- Theta - eta . Adam(grad_Theta L_total)
    9:   Evaluate L_total on validation fold; record best Theta* if improved
    10:  if no improvement for 15 epochs: break (early stopping)
    11: return Theta*
"""

import copy
from typing import Dict, Iterator, Tuple

import torch
from torch.utils.data import DataLoader

from .config import QuantKANEVConfig
from .model import QuantKANEV
from .losses import total_loss


def xavier_initialize(model: QuantKANEV) -> None:
    """
    Algorithm 1, line 1: Xavier-initialize all Linear layers. The
    rotation-gate parameters (alpha_k <- 1, beta_k <- 0) and coupling
    matrices (W <- I, b <- 0) are already initialized this way in
    `QuantumInspiredEdgeActivation.__init__` (activation.py).
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


def run_epoch(
    model: QuantKANEV,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: QuantKANEVConfig,
    train: bool = True,
) -> Dict[str, float]:
    """
    One pass over `loader`. Each batch is expected to yield
    (x_window, c_t, soh_true, c_true_horizon):
        x_window : (B, w, d')
        c_t      : (B,)          current capacity, used as fade-forecast anchor
        soh_true : (B,)          ground-truth SoH in [0,1]
        c_true_horizon : (B, H)  ground-truth capacity trajectory
    """
    model.train(mode=train)
    running = {"L_SoH": 0.0, "L_fade": 0.0, "L_reg": 0.0, "L_total": 0.0}
    n_batches = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x_window, c_t, soh_true, c_true_horizon in loader:
            soh_hat, c_forecast = model(x_window, c_t)

            loss, components = total_loss(
                model,
                soh_hat,
                soh_true,
                c_forecast,
                c_true_horizon,
                loss_weights=config.loss_weights,
                rho1=config.reg_rho1,
                rho2=config.reg_rho2,
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            for k, v in components.items():
                running[k] += v
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}


def train_source_domain(
    model: QuantKANEV,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: QuantKANEVConfig,
) -> Tuple[QuantKANEV, Dict[str, list]]:
    """
    Full Algorithm 1 training loop with early stopping.

    Returns
    -------
    best_model : QuantKANEV with the best validation-loss weights (Theta*)
    history : dict of per-epoch train/val loss traces, for diagnostic plots
              analogous to Figure 6 (training/validation loss curves).
    """
    xavier_initialize(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.source_lr)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, config.max_epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, config, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, config, train=False)

        history["train_loss"].append(train_metrics["L_total"])
        history["val_loss"].append(val_metrics["L_total"])

        if val_metrics["L_total"] < best_val_loss:
            best_val_loss = val_metrics["L_total"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"[epoch {epoch:03d}] train_loss={train_metrics['L_total']:.6f} "
            f"val_loss={val_metrics['L_total']:.6f} "
            f"(best={best_val_loss:.6f}, no-improve={epochs_without_improvement})"
        )

        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch} (Algorithm 1, line 10).")
            break

    model.load_state_dict(best_state)
    return model, history
