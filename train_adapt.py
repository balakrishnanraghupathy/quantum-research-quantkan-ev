"""
train_adapt.py
===============
Algorithm 2 -- Cross-Domain Selective Adaptation (Section 3.8).

    Input: Source-trained Theta*; target dataset D_t; K rolling-origin
           folds; adaptation learning rate eta_a
    Output: Adapted parameters Theta_adapted, per-fold metrics

    1: for fold k = 1 to K do
    2:   Align target features via Eq. 19; re-calibrate fusion-path knots
    3:   Freeze rotation-angle parameters of layers 1..L-1
    4:   Fine-tune coupling/output-mixing weights, final layer & heads
        on training folds
    5:   Evaluate on the held-out fold; record Accuracy, R2, RMSE, MAE
    6: return Theta_adapted, per-fold metrics (aggregated in Table 16)

Fold boundaries follow the ordering of vehicles and cycles (rolling-
origin), so later information never leaks into earlier predictions
(Section 3.8).
"""

import copy
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import QuantKANEVConfig
from .model import QuantKANEV
from .losses import total_loss
from .domain_adapt import (
    apply_selective_freezing,
    align_features,
    compute_domain_statistics,
)
from .metrics import evaluate_all


def rolling_origin_folds(num_windows: int, num_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Constructs K rolling-origin (expanding-window) train/eval index
    splits over a chronologically-ordered sequence of `num_windows`
    target-domain samples, so that for fold k, all evaluation indices
    occur strictly after all training indices for that fold -- this
    prevents later information from leaking into earlier predictions
    (Section 3.8).

    NOTE: callers are responsible for ensuring `num_windows` indices are
    already sorted by (vehicle, cycle) order before calling this
    function, per the manuscript's fold-construction rule.
    """
    fold_edges = np.linspace(0, num_windows, num_folds + 1, dtype=int)
    folds = []
    for k in range(1, num_folds + 1):
        train_idx = np.arange(0, fold_edges[k - 1] if k > 1 else fold_edges[1] // 2)
        eval_start, eval_end = fold_edges[k - 1], fold_edges[k]
        eval_idx = np.arange(max(eval_start, train_idx[-1] + 1 if len(train_idx) else 0), eval_end)
        if len(train_idx) == 0 or len(eval_idx) == 0:
            continue
        folds.append((train_idx, eval_idx))
    return folds


def adapt_one_fold(
    source_model: QuantKANEV,
    fold_train_loader: DataLoader,
    fold_eval_loader: DataLoader,
    source_stats: Tuple[torch.Tensor, torch.Tensor],
    target_stats: Tuple[torch.Tensor, torch.Tensor],
    config: QuantKANEVConfig,
    adapt_epochs: int = 30,
) -> Tuple[QuantKANEV, Dict[str, float]]:
    """
    Runs the selective fine-tuning procedure (Algorithm 2, steps 2-5) on
    a single rolling-origin fold, starting from a deep copy of the
    source-pretrained model.
    """
    model = copy.deepcopy(source_model)
    mu_s, sigma_s = source_stats
    mu_t, sigma_t = target_stats

    # Step 3: freeze early layers; leave final layer + heads trainable.
    trainable_params = apply_selective_freezing(model)

    # Adaptation learning rate is 1/10 of the source learning rate (Table 8).
    optimizer = torch.optim.Adam(trainable_params, lr=config.adapt_lr)

    # Step 4: fine-tune on the fold's training portion.
    model.train()
    for _ in range(adapt_epochs):
        for x_window, c_t, soh_true, c_true_horizon in fold_train_loader:
            # Step 2: align target features to the source domain's scale (Eq. 19).
            b, w, d = x_window.shape
            x_flat = x_window.reshape(b * w, d)
            x_aligned = align_features(x_flat, mu_s, sigma_s, mu_t, sigma_t)
            x_window_aligned = x_aligned.reshape(b, w, d)

            soh_hat, c_forecast = model(x_window_aligned, c_t)
            loss, _ = total_loss(
                model,
                soh_hat,
                soh_true,
                c_forecast,
                c_true_horizon,
                loss_weights=config.loss_weights,
                rho1=config.reg_rho1,
                rho2=config.reg_rho2,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Step 5: evaluate on the held-out portion of this fold.
    model.eval()
    all_soh_hat, all_soh_true = [], []
    with torch.no_grad():
        for x_window, c_t, soh_true, _ in fold_eval_loader:
            b, w, d = x_window.shape
            x_flat = x_window.reshape(b * w, d)
            x_aligned = align_features(x_flat, mu_s, sigma_s, mu_t, sigma_t)
            x_window_aligned = x_aligned.reshape(b, w, d)

            soh_hat, _ = model(x_window_aligned, c_t)
            all_soh_hat.append(soh_hat.cpu().numpy())
            all_soh_true.append(soh_true.cpu().numpy())

    y_hat = np.concatenate(all_soh_hat)
    y_true = np.concatenate(all_soh_true)
    fold_metrics = evaluate_all(y_hat, y_true)
    return model, fold_metrics


def run_cross_domain_adaptation(
    source_model: QuantKANEV,
    target_fold_loaders: List[Tuple[DataLoader, DataLoader]],
    source_domain_features: torch.Tensor,
    target_domain_features: torch.Tensor,
    config: QuantKANEVConfig,
) -> Tuple[List[QuantKANEV], List[Dict[str, float]]]:
    """
    Full Algorithm 2 loop across K rolling-origin folds (Table 16's
    "Selective cross-domain adaptation" row is the aggregate of these
    per-fold results).

    Parameters
    ----------
    target_fold_loaders : list of (train_loader, eval_loader) pairs, one
        per rolling-origin fold, already constructed via
        `rolling_origin_folds` and wrapped in DataLoaders by the caller.
    source_domain_features, target_domain_features : Tensor (N, d') used
        to compute (mu_s, sigma_s) and (mu_t, sigma_t) for Eq. 19.
    """
    source_stats = compute_domain_statistics(source_domain_features)
    target_stats = compute_domain_statistics(target_domain_features)

    fold_models, fold_metrics = [], []
    for k, (train_loader, eval_loader) in enumerate(target_fold_loaders, start=1):
        print(f"--- Cross-domain adaptation: fold {k}/{len(target_fold_loaders)} ---")
        model_k, metrics_k = adapt_one_fold(
            source_model, train_loader, eval_loader, source_stats, target_stats, config
        )
        fold_models.append(model_k)
        fold_metrics.append(metrics_k)
        print(f"  fold {k} metrics: {metrics_k}")

    return fold_models, fold_metrics


def aggregate_fold_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    """
    Returns (mean, std) per metric across folds, analogous to the
    fold-wise summary reported in Figure 15 (97.72%-98.21%, mean 97.95%
    +/- 0.19 s.d.).
    """
    keys = fold_metrics[0].keys()
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics])
        out[k] = (float(vals.mean()), float(vals.std()))
    return out
