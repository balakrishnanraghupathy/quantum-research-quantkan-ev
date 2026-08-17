"""
example_run.py
===============
End-to-end demonstration of the QuantKAN-EV reference implementation on
SYNTHETIC data, showing: model construction, a source-domain training
loop (Algorithm 1), the guaranteed-monotonicity check for the capacity-
fade head (Eq. 18 proof), cross-domain selective adaptation (Algorithm
2), and evaluation metrics (Eq. 20-23).

This script uses randomly generated data purely to exercise every code
path end-to-end and confirm the implementation is runnable and
internally consistent with the equations/algorithms it claims to
implement. It is NOT a reproduction of the manuscript's reported
results, which require the actual NASA, CALCE, and VIT-EV-battery
datasets (Section 3.2) -- see README.md, "Reproducing paper results."
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from quantkanev.config import QuantKANEVConfig
from quantkanev.model import QuantKANEV
from quantkanev.train_source import train_source_domain
from quantkanev.train_adapt import (
    rolling_origin_folds,
    run_cross_domain_adaptation,
    aggregate_fold_metrics,
)
from quantkanev.capacity_fade import verify_monotonicity_guarantee
from quantkanev.metrics import evaluate_all, macro_average


def make_synthetic_dataset(n_samples: int, config: QuantKANEVConfig) -> TensorDataset:
    """Generates a synthetic dataset matching the model's expected input shapes."""
    x_window = torch.randn(n_samples, config.window_length, config.input_dim)
    c_t = torch.rand(n_samples) * 0.5 + 0.5           # current capacity in (0.5, 1.0)
    soh_true = torch.rand(n_samples)                    # ground-truth SoH in [0, 1]

    # Synthetic ground-truth capacity trajectory: strictly non-increasing,
    # matching the physical prior the model is designed to respect.
    decrements = torch.rand(n_samples, config.forecast_horizon) * 0.01
    c_true_horizon = c_t.unsqueeze(-1) - torch.cumsum(decrements, dim=-1)

    return TensorDataset(x_window, c_t, soh_true, c_true_horizon)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    config = QuantKANEVConfig()
    print("=== QuantKAN-EV: configuration (Table 8) ===")
    print(config)

    # ------------------------------------------------------------------
    # 1. Model construction and parameter count sanity check (Table 18:
    #    135,979 parameters reported for the full configuration).
    # ------------------------------------------------------------------
    model = QuantKANEV(config)
    n_params = model.count_parameters()
    print(f"\nTotal trainable parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # 2. Guaranteed-monotonicity verification (Eq. 18 proof), using an
    #    UNTRAINED head -- the guarantee should hold regardless.
    # ------------------------------------------------------------------
    holds = verify_monotonicity_guarantee(model.fade_head)
    print(f"Monotonicity guarantee holds on untrained head: {holds}")
    assert holds, "Monotonicity guarantee failed -- check capacity_fade.py"

    # ------------------------------------------------------------------
    # 3. Source-domain training (Algorithm 1), on synthetic source data.
    # ------------------------------------------------------------------
    print("\n=== Source-domain training (Algorithm 1, synthetic data) ===")
    source_train_ds = make_synthetic_dataset(256, config)
    source_val_ds = make_synthetic_dataset(64, config)

    # Small epoch budget for a quick demonstration run.
    config.max_epochs = 3
    config.early_stopping_patience = 3

    train_loader = DataLoader(source_train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(source_val_ds, batch_size=config.batch_size, shuffle=False)

    model, history = train_source_domain(model, train_loader, val_loader, config)

    # ------------------------------------------------------------------
    # 4. Source-domain evaluation metrics (Eq. 20-23), analogous to
    #    Table 15.
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        x_window, c_t, soh_true, _ = source_val_ds.tensors
        soh_hat, _ = model(x_window, c_t)
    source_metrics = evaluate_all(soh_hat.numpy(), soh_true.numpy())
    print("\nSource-domain (synthetic) validation metrics (cf. Table 15):")
    for k, v in source_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ------------------------------------------------------------------
    # 5. Cross-domain adaptation (Algorithm 2), on synthetic target data,
    #    K=5 rolling-origin folds.
    # ------------------------------------------------------------------
    print("\n=== Cross-domain adaptation (Algorithm 2, synthetic EV data) ===")
    n_target = 500
    target_ds = make_synthetic_dataset(n_target, config)
    folds = rolling_origin_folds(n_target, config.num_folds)

    fold_loaders = []
    for train_idx, eval_idx in folds:
        train_subset = TensorDataset(*[t[train_idx] for t in target_ds.tensors])
        eval_subset = TensorDataset(*[t[eval_idx] for t in target_ds.tensors])
        fold_loaders.append(
            (
                DataLoader(train_subset, batch_size=config.batch_size, shuffle=True),
                DataLoader(eval_subset, batch_size=config.batch_size, shuffle=False),
            )
        )

    source_features = source_train_ds.tensors[0].mean(dim=1)  # (N, d') pooled over window
    target_features = target_ds.tensors[0].mean(dim=1)

    _, fold_metrics = run_cross_domain_adaptation(
        model, fold_loaders, source_features, target_features, config
    )
    agg = aggregate_fold_metrics(fold_metrics)
    print("\nCross-domain adaptation, aggregated across folds (cf. Table 16 / Figure 15):")
    for k, (mean, std) in agg.items():
        print(f"  {k}: {mean:.4f} +/- {std:.4f}")

    # ------------------------------------------------------------------
    # 6. Headline macro-average across all three test sets (cf. Section
    #    4.5's Accuracy = mean(NASA, CALCE, EV) = 98.63% derivation).
    # ------------------------------------------------------------------
    ev_metrics = {k: v[0] for k, v in agg.items()}
    headline = macro_average(source_metrics, source_metrics, ev_metrics)
    print("\nIllustrative macro-average across test sets (cf. Section 4.5):")
    for k, v in headline.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
