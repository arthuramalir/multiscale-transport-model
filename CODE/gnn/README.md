# GNN for Pore-Network Transport Prediction

This module implements a **Graph Neural Network (GNN)** surrogate that predicts
the effective diffusivity `D_eff` of a pore network directly from its graph
structure — without running the full finite-volume solver.

Pore networks are naturally represented as graphs:
- **Nodes** = pores (3D position, diameter, volume)
- **Edges** = throats (diameter, length, diffusive conductance)
- **Target** = log₁₀(D_eff) — a graph-level scalar

---

## Architecture

```
Input graph (variable Np nodes, Nt edges)
        │
        ▼
 Edge encoder MLP  (3 → 128)          ← throat features
        │
        ▼  aggregated to nodes
 Input projection  (5+128 → 128)      ← pore + edge-aggregated features
        │
        ▼
 ×3  SAGEConv + BatchNorm + ReLU      ← message passing (residual)
        │
        ▼
 Global mean-pool ⊕ max-pool          ← fixed-size graph embedding (256)
        │
 Graph-level features (L, ρ) → 128   ← domain size + pore density
        │ concatenate
        ▼
 MLP: 384 → 128 → 64 → 1             ← regression head
```

**Trainable parameters:** ~200 k (default `hidden=128`)

---

## Files

| File | Purpose |
|------|---------|
| `data_pipeline.py` | Build PyTorch Geometric `Data` objects from sweep CSVs; cache `.pt` files |
| `model.py` | `PoreNetGNN` model definition |
| `train.py` | Full training script with evaluation, plots, and Ridge baseline |
| `cache/` | Auto-created; stores per-sample `.pt` graph files |
| `results/` | Auto-created; stores model weights, plots, and JSON summary |

---

## Installation

```bash
pip install torch>=2.6.0 torch_geometric>=2.6.1 scikit-learn>=1.6.0
# or simply:
pip install -r requirements.txt
```

---

## Quick Start

### 1. Build the graph dataset (requires OpenPNM)

Networks are deterministically regenerated from their random seeds.
Results are cached as `.pt` files so this step only runs once.

```bash
python -m CODE.gnn.data_pipeline \
    --csv "CODE/FINAL SET/const_density_1e12/sweep_const_density_1e12_full.csv" \
    --cache CODE/gnn/cache/1e12
```

### 2. Train the GNN

```bash
python -m CODE.gnn.train \
    --csv "CODE/FINAL SET/const_density_1e12/sweep_const_density_1e12_full.csv" \
    --cache CODE/gnn/cache/1e12 \
    --out   CODE/gnn/results/1e12 \
    --epochs 150 \
    --batch-size 32 \
    --hidden 128 \
    --lr 3e-4
```

Add `--max-samples 200` to do a quick smoke-test on a subset.

### Latest smoke-test benchmark

Run on 2026-03-25 with `--max-samples 200`, `--epochs 150`:

- Split: train=140, val=30, test=30
- Model parameters: 191,617
- GNN test: R2=0.9526, MAE=0.1751, RMSE=0.2439
- Ridge baseline test: R2=0.6506, MAE=0.5138, RMSE=0.6625
- Delta R2 (GNN - Ridge): +0.3020

Artifacts were saved to `CODE/gnn/results/1e12/`.

### 3. Inspect outputs

After training, `CODE/gnn/results/1e12/` will contain:

| File | Description |
|------|-------------|
| `best_model.pt` | Best model weights (by validation loss) |
| `training_curves.png` | Train / val MSE loss per epoch |
| `parity_plot.png` | Predicted vs actual log₁₀(D_eff) |
| `residuals_hist.png` | Error distribution + residuals vs actual |
| `feature_importance.png` | Ridge regression baseline coefficients |
| `results_summary.json` | R², MAE, RMSE for GNN and Ridge baseline |

---

## Node & Edge Features

### Node features (5 dimensions)

| # | Feature | Notes |
|---|---------|-------|
| 0 | log₁₀(d_pore / 1 µm) | Log-scaled pore diameter |
| 1 | log₁₀(V_pore / 1 µm³) | Log-scaled pore volume |
| 2 | x / L | Normalised x-coordinate |
| 3 | y / L | Normalised y-coordinate |
| 4 | z / L | Normalised z-coordinate |

### Edge features (3 dimensions per direction)

| # | Feature | Notes |
|---|---------|-------|
| 0 | log₁₀(d_throat / 1 µm) | Log-scaled throat diameter |
| 1 | log₁₀(l_throat / 1 µm) | Log-scaled throat length |
| 2 | log₁₀(g / 1e-20) | Log-scaled diffusive conductance D·A/l |

Edges are stored bidirectionally (both directions for undirected throats).

### Graph-level features (2 dimensions)

| # | Feature |
|---|---------|
| 0 | log₁₀(L / 1 µm) — domain size |
| 1 | log₁₀(pore density in m⁻³) |

---

## Baseline Comparison

The training script automatically trains a **Ridge regression** model on
per-graph summary statistics (feature means) as a classical ML baseline.
The parity plot includes both R² scores so you can see the benefit of the
full graph representation over simple aggregated descriptors.

---

## Python API

```python
import torch
from CODE.gnn.data_pipeline import build_dataset, dataset_stats, normalize_dataset
from CODE.gnn.model import build_model

# Build dataset
dataset = build_dataset(
    csv_path="CODE/FINAL SET/const_density_1e12/sweep_const_density_1e12_full.csv",
    cache_dir="CODE/gnn/cache/1e12",
    max_samples=100,
)

# Normalise
stats = dataset_stats(dataset)
normed = normalize_dataset(dataset, stats)

# Build model
model = build_model(stats, hidden=128)
print(f"Parameters: {model.count_parameters():,}")

# Load trained weights and predict
model.load_state_dict(torch.load("CODE/gnn/results/1e12/best_model.pt", weights_only=True))
model.eval()
with torch.no_grad():
    pred_norm = model(normed[0])
    pred_log_deff = pred_norm.item() * stats["target_std"] + stats["target_mean"]
    print(f"Predicted log₁₀(D_eff) = {pred_log_deff:.3f}")
    print(f"Predicted D_eff = {10**pred_log_deff:.3e} m²/s")
```

---

## Design Notes

- **Log-scaling** of all size quantities (diameters, volumes, conductances)
  makes the input distributions closer to Gaussian and improves training
  stability.
- **Residual connections** in the SAGEConv stack prevent gradient vanishing
  for deeper architectures.
- **Mean + max pooling** captures both average network behaviour and
  worst-case bottlenecks (the minimum-conductance throat often controls D_eff).
- **Graph-level features** (L, density) are injected at the readout stage
  rather than added to every node; this avoids diluting local structural
  signals with global information.
- The **Ridge baseline** uses only per-graph mean feature values, deliberately
  discarding structural information, so any R² improvement of the GNN
  directly quantifies the value of graph topology.
