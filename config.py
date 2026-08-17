"""
config.py
=========
Principal hyperparameters of the proposed QuantKAN-EV framework,
reproduced from Table 8 of the manuscript.

All defaults below correspond exactly to the values used to obtain the
results reported in Section 4 (Tables 15, 16, 18, 19, 20, 21). The
`search_range` fields document the range each hyperparameter was tuned
over on the held-out source-domain validation split (Section 3.10),
not the final chosen value.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class QuantKANEVConfig:
    # ---- Windowing / forecast horizon --------------------------------
    window_length: int = 30           # w   (search range 10-60 cycles)
    forecast_horizon: int = 10        # H   (search range 5-20 cycles)

    # ---- Backbone architecture (Table 5) ------------------------------
    input_dim: int = 5                # d' = d(4 raw channels) + 1 (DeltaCapacity)
    hidden_dim: int = 64               # QuantKAN layer 1 width
    layer2_dim: int = 48                # QuantKAN layer 2 width
    fusion_out_dim: int = 32            # Adaptive Spline Fusion output width
    num_quantkan_layers: int = 2        # L   (+1 fusion layer) (search range 1-4)

    # ---- Quantum-inspired activation unit (Eq. 5-9) --------------------
    num_rotation_gates: int = 3          # Q   (search range 1-5)
    spline_grid_size: int = 8            # G   (search range 4-16)
    spline_basis_terms: int = 8          # number of B-spline coefficients c_i (Eq. 4, 11)

    # ---- Optimization ---------------------------------------------------
    source_lr: float = 1e-3              # eta        (search range 1e-4 - 5e-3)
    adapt_lr: float = 1e-4               # eta_a      (search range 1e-5 - 1e-3), = 1/10 of source_lr
    batch_size: int = 64                 # B          (search range 16-128)
    num_folds: int = 5                   # K          (source 85/15 val split; target K=5 rolling-origin)
    max_epochs: int = 150
    early_stopping_patience: int = 15    # epochs without improvement (Algorithm 1, line 10)

    # ---- Loss weights (Eq. 16): lambda_1, lambda_2, lambda_3 -----------
    loss_weights: Tuple[float, float, float] = (1.0, 0.7, 0.01)

    # ---- Regularization coefficients (Eq. 15): rho_1, rho_2 ------------
    reg_rho1: float = 1e-3               # penalizes rotation-gate (alpha, beta) magnitude
    reg_rho2: float = 1e-4               # penalizes residual spline coefficients c_{q,i}

    # ---- Source-domain train/val/test split (Table 14) -----------------
    source_train_frac: float = 0.70
    source_val_frac: float = 0.15         # of the 70% training partition -> 85/15 internal split
    source_test_frac: float = 0.15

    # ---- Data channel names (Section 3.2, Stage I/II) -------------------
    raw_channels: Tuple[str, ...] = field(
        default_factory=lambda: ("voltage", "current", "temperature", "soc_or_soh")
    )
    derived_channels: Tuple[str, ...] = field(default_factory=lambda: ("delta_capacity",))

    def __post_init__(self):
        assert self.input_dim == len(self.raw_channels) + len(self.derived_channels), (
            "input_dim must equal number of raw channels + derived channels (d' = d + 1, Section 3.2)"
        )
        assert len(self.loss_weights) == 3, "loss_weights must be (lambda_1, lambda_2, lambda_3)"
