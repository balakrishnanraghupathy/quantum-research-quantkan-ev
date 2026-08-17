# QuantKAN-EV: Reference Implementation

A quantum-inspired Kolmogorov-Arnold Network for cross-domain State of
Health (SoH) estimation and capacity-fade prediction in electric
vehicle lithium-ion batteries. This package implements the architecture,
training procedure, and cross-domain adaptation strategy described in
Sections 3.1–3.10 of the manuscript, mapped equation-by-equation and
algorithm-by-algorithm to code.

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```bash
python example_run.py
```

This runs the full pipeline end-to-end on **synthetic** data: model
construction, a short source-domain training loop (Algorithm 1), a
runtime check of the guaranteed-monotonicity proof (Eq. 18), K=5
rolling-origin cross-domain adaptation (Algorithm 2), and evaluation
metrics (Eq. 20–23). It is meant to confirm the implementation is
runnable and internally consistent — **it does not reproduce the
manuscript's reported numerical results** (Tables 15, 16, 18–21), which
require the actual NASA, CALCE, and VIT-EV-battery datasets. See
"Reproducing paper results" below.

## Running the tests

```bash
python tests/test_core.py
# or, if pytest is installed:
pytest tests/ -v
```

The test suite checks: edge-activation output shapes and parameter
counts, the Eq. 9 reduction limit, the guaranteed-monotonicity proof
(Section 3.7) under both an untrained head and a perturbed/post-training-step
head, the diagonal-coupling ablation (Table 19, variant b), full-model
forward-pass shapes, and the metric formulas (Eq. 20–23).

## Package structure

```
quantkanev/
├── __init__.py
├── config.py          Hyperparameters (Table 8)
├── activation.py       Quantum-inspired edge activation psi_{q,p} (Eq. 5-9),
│                        reference single-edge implementation + reduction-limit
│                        and diagonal-coupling helpers
├── layers.py            Vectorized QuantKAN layer (Eq. 3, 4, 11, 12) --
│                        the implementation actually used inside the model,
│                        for tractable speed at Table 5's widths (up to 4096
│                        edges/layer)
├── model.py             Full QuantKAN-EV backbone + dual-head module (Table 5)
├── capacity_fade.py      Monotonic capacity-fade head (Eq. 17-18) + proof
│                        verification utility
├── losses.py             Multi-task loss (Eq. 13-16)
├── metrics.py             Evaluation metrics (Eq. 20-23)
├── domain_adapt.py        Feature-statistics alignment (Eq. 19) + selective
│                        freezing (Algorithm 2)
├── train_source.py        Algorithm 1: source-domain training loop
├── train_adapt.py          Algorithm 2: cross-domain adaptation loop,
│                        rolling-origin fold construction
└── data_pipeline.py        Stage II preprocessing (Section 3.2): cleaning,
                          imputation, outlier rejection, normalization,
                          DeltaCapacity, windowing

example_run.py           End-to-end demonstration on synthetic data
tests/test_core.py        Unit tests / sanity checks
requirements.txt
```

## Equation-to-code map

| Manuscript | Code |
|---|---|
| Eq. 3 (Kolmogorov-Arnold representation) | `layers.py: QuantKANLayer.forward` (aggregation structure) |
| Eq. 4 (standard KAN edge / B-spline basis) | `layers.py: _bspline_basis`; see also the Eq. 9 reduction limit |
| Eq. 5–8 (quantum-inspired activation) | `activation.py: QuantumInspiredEdgeActivation`; vectorized in `layers.py: QuantKANLayer.forward` |
| Eq. 9 (reduction limit) | `activation.py: set_reduction_limit`; `layers.py: QuantKANLayer.set_reduction_limit` |
| Eq. 10 (coupling-gap term) | `layers.py: QuantKANLayer(diagonal_coupling=True)` sets this to zero (ablation variant b) |
| Eq. 11–12 (layer aggregation + stacking) | `layers.py: QuantKANLayer.forward`; `model.py: QuantKANEV.backbone` |
| Table 5 (backbone config) | `model.py: QuantKANEV.__init__` |
| Eq. 13–16 (multi-task loss) | `losses.py` |
| Algorithm 1 (source training) | `train_source.py: train_source_domain` |
| Eq. 17–18 (capacity-fade forecast + monotonicity) | `capacity_fade.py: CapacityFadeHead` |
| Eq. 19 (domain-statistics alignment) | `domain_adapt.py: align_features` |
| Algorithm 2 (cross-domain adaptation) | `train_adapt.py: run_cross_domain_adaptation`; `domain_adapt.py: apply_selective_freezing` |
| Eq. 20–23 (evaluation metrics) | `metrics.py` |
| Table 8 (hyperparameters) | `config.py: QuantKANEVConfig` |
| Table 14 (dataset splits) | `config.py`: `source_train_frac` / `source_val_frac` / `source_test_frac`; `train_adapt.py: rolling_origin_folds` for the K=5 target-domain folds |
| Table 19 ablation variant (a), "w/o rotation gates" | `layers.py: QuantKANLayer.set_reduction_limit()` |
| Table 19 ablation variant (b), "w/o entanglement coupling W" | `layers.py: QuantKANLayer(diagonal_coupling=True)` |
| Table 19 ablation variant (c), "w/o adaptive spline fusion path" | Construct `QuantKANEV` and skip/bypass `self.fusion` in a subclassed `backbone()` |
| Table 19 ablation variant (d), "w/o dual-head parameter sharing" | Instantiate two independent `QuantKANEV`-style backbones, one per head, instead of sharing `h_L` |
| Table 19 ablation variant (e), "w/o selective freezing" | Skip `domain_adapt.apply_selective_freezing`; fine-tune all parameters directly in `train_adapt.py` |

## Known implementation choices / deviations worth disclosing

- **B-spline basis**: `layers.py: _bspline_basis` uses a bounded cosine
  basis over a tanh-squashed input as a numerically stable stand-in for
  a true B-spline basis fit on a fixed grid of G knots (Table 8). Any
  proper B-spline implementation (e.g. `scipy.interpolate.BSpline`) can
  be substituted without changing the surrounding aggregation logic.
  Because of this substitution, the exact trainable-parameter count
  reported by `QuantKANEV.count_parameters()` will not precisely match
  the manuscript's reported 135,979 (Table 18) unless the residual
  spline path is reimplemented to match exactly; the quantum-inspired
  edge-activation parameter count (3Q+Q² per edge) does match exactly.
- **QuantKANLayer vectorization**: `activation.py`'s
  `QuantumInspiredEdgeActivation` is a small, readable, mathematically
  exact single-edge reference module. `layers.py`'s `QuantKANLayer`
  stores the same per-edge parameters as batched tensors and computes
  all edges in parallel for tractable training speed at the widths used
  in Table 5 (a naive per-edge `nn.Module` loop is computationally
  infeasible at 64×64 = 4096 edges/layer). The two are mathematically
  identical per edge.
- **ΔCapacity computation for the EV domain**: `data_pipeline.py:
  compute_delta_capacity()` implements the `"diff"` method (direct
  cycle-to-cycle difference of a measured reference capacity), suitable
  for the NASA/CALCE lab datasets. The manuscript does not specify how
  ΔCapacity should be computed for the field-collected EV domain (from
  BMS-reported SoH vs. partial coulomb counting); this is flagged
  explicitly in both the manuscript's Future Directions (Section 5) and
  in this function's docstring, and is left unimplemented (raises
  `NotImplementedError`) pending that characterization, rather than
  silently assuming one method.
- **Floating-point precision of the monotonicity guarantee**: see the
  note in `capacity_fade.py`'s module docstring — the guarantee is exact
  in real arithmetic but can encounter float32 underflow at
  pathologically large (untrained-scale) parameter magnitudes; switching
  to float64 removes this entirely if relevant to a given deployment.
- **On-device / quantized deployment, uncertainty quantification, and
  out-of-distribution detection** are not implemented in this package,
  consistent with these being named as open items in the manuscript's
  Future Directions (Section 5) rather than demonstrated results.

## Reproducing paper results

The manuscript's reported numbers (Tables 15, 16, 18–21) require the
actual datasets:

- **NASA Li-Ion Battery Aging dataset**: https://data.nasa.gov/dataset/li-ion-battery-aging-datasets
- **CALCE battery data**: https://calce.umd.edu/battery-data
- **Real-world two-wheeler EV dataset (VIT-EV-battery)**: obtained from
  Vellore Institute of Technology (VIT), Vellore, India, under
  institutional permission for research use only (see the manuscript's
  Data Availability Statement for access terms).

To reproduce Table 15/16: run `data_pipeline.preprocess_pipeline()` on
each raw dataset, feed the resulting windowed tensors through
`train_source.train_source_domain()` (Algorithm 1) for NASA+CALCE, then
`train_adapt.run_cross_domain_adaptation()` (Algorithm 2) for the EV
target domain, using the exact hyperparameters in `config.py`
(reproduced from Table 8).

## Citation

If you use this code, please cite the associated manuscript (full
citation to be inserted upon acceptance/publication).
