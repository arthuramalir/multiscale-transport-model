"""train.py — Train and evaluate the PoreNetGNN model.

Workflow
--------
1. Build (or load from cache) the PyG graph dataset from a sweep CSV.
2. Split into train / validation / test (70 / 15 / 15 %).
3. Normalise node features, edge features, and the target.
4. Train the GNN with AdamW + cosine annealing LR schedule.
5. Evaluate on the test set: MAE, RMSE, R².
6. Compare against a Ridge-regression baseline using per-graph summary statistics.
7. Save plots:
     - training_curves.png      (train/val loss vs epoch)
     - parity_plot.png          (predicted vs actual log10(D_eff))
     - residuals_hist.png       (error distribution)
     - feature_importance.png   (baseline Ridge coefficients)
8. Save the trained model weights to  <out_dir>/best_model.pt

Usage
-----
    python -m CODE.gnn.train \\
        --csv "CODE/FINAL SET/const_density_1e12/sweep_const_density_1e12_full.csv" \\
        --cache CODE/gnn/cache/1e12 \\
        --out   CODE/gnn/results/1e12 \\
        --epochs 150 \\
        --batch-size 32 \\
        --hidden 128 \\
        --lr 3e-4

All paths are relative to the working directory from which the script is called.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Lazy imports (torch / sklearn) with helpful error messages
# ---------------------------------------------------------------------------

def _import_torch():
    try:
        import torch
        from torch_geometric.data import DataLoader  # noqa: F401
        return torch
    except ImportError as e:
        sys.exit(
            f"ERROR: {e}\n"
            "Install dependencies:\n"
            "  pip install torch torch_geometric scikit-learn matplotlib\n"
        )


def _import_sklearn():
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score
        return Ridge, StandardScaler, r2_score
    except ImportError:
        return None, None, None


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = _import_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _train_val_test_split(
    dataset: List, seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15
) -> Tuple[List, List, List]:
    """Shuffle and split dataset into train/val/test."""
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    n_test = max(1, int(len(indices) * test_frac))
    n_val = max(1, int(len(indices) * val_frac))
    test_idx = indices[:n_test]
    val_idx = indices[n_test: n_test + n_val]
    train_idx = indices[n_test + n_val:]

    return (
        [dataset[i] for i in train_idx],
        [dataset[i] for i in val_idx],
        [dataset[i] for i in test_idx],
    )


def _summary_features(dataset: List) -> np.ndarray:
    """Extract per-graph summary statistics for the Ridge baseline.

    Features: [mean_x_dim0, ..., mean_x_dim4,
               mean_edge_dim0, mean_edge_dim1, mean_edge_dim2,
               graph_feat_0, graph_feat_1,
               log_num_nodes, log_num_edges]
    """
    rows = []
    for d in dataset:
        x_mean = d.x.mean(0).numpy()            # (5,)
        e_mean = d.edge_attr.mean(0).numpy()    # (3,)
        gf = d.graph_feat.view(-1).numpy()      # (2,)
        ln = math.log10(max(d.num_nodes, 1))
        le = math.log10(max(d.edge_attr.shape[0], 1))
        rows.append(np.concatenate([x_mean, e_mean, gf, [ln, le]]))
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, optimizer, device, train: bool):
    """Single train or eval pass. Returns mean MSE loss."""
    import torch
    import torch.nn.functional as F

    model.train(train)
    total_loss = 0.0
    total_samples = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).squeeze(-1)   # (B,)
            loss = F.mse_loss(pred, batch.y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            bs = batch.y.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

    return total_loss / max(total_samples, 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(model, dataset: List, stats: dict, device, batch_size: int = 64):
    """Return (predictions, targets) in the *original* log10(D_eff) scale."""
    import torch
    from torch_geometric.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, targets = [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_norm = model(batch).squeeze(-1).cpu()   # normalised
            y_norm = batch.y.cpu()
            # Denormalise
            p_orig = pred_norm * stats["target_std"] + stats["target_mean"]
            y_orig = y_norm * stats["target_std"] + stats["target_mean"]
            preds.append(p_orig.numpy())
            targets.append(y_orig.numpy())

    return np.concatenate(preds), np.concatenate(targets)


def _metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-40))
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_training_curves(train_losses, val_losses, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train MSE (normalised)")
    ax.plot(epochs, val_losses, label="Val MSE (normalised)")
    best_ep = int(np.argmin(val_losses)) + 1
    ax.axvline(best_ep, color="red", linestyle="--", alpha=0.6, label=f"Best val (ep {best_ep})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (normalised target)")
    ax.set_title("GNN Training Curves — PoreNetGNN")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_parity(preds, targets, gnn_metrics: dict, baseline_metrics: dict | None,
                 out_dir: str, label: str = "GNN") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(targets, preds, alpha=0.4, s=20, label=f"{label} predictions")
    lo = min(targets.min(), preds.min()) - 0.3
    hi = max(targets.max(), preds.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="Perfect prediction")
    if baseline_metrics:
        ax.set_title(
            f"Parity plot — log₁₀(D_eff)\n"
            f"GNN  R²={gnn_metrics['r2']:.3f}, MAE={gnn_metrics['mae']:.3f}\n"
            f"Ridge R²={baseline_metrics['r2']:.3f}, MAE={baseline_metrics['mae']:.3f}"
        )
    else:
        ax.set_title(
            f"Parity plot — log₁₀(D_eff)\n"
            f"R²={gnn_metrics['r2']:.3f}, MAE={gnn_metrics['mae']:.3f}"
        )
    ax.set_xlabel("Actual log₁₀(D_eff)")
    ax.set_ylabel("Predicted log₁₀(D_eff)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(out_dir, "parity_plot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_residuals(preds, targets, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    residuals = preds - targets
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(residuals, bins=40, edgecolor="k", alpha=0.75)
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Residual (predicted − actual)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Residuals — std={residuals.std():.3f}")

    axes[1].scatter(targets, residuals, alpha=0.4, s=20)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Actual log₁₀(D_eff)")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs Actual")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(out_dir, "residuals_hist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_baseline_coefficients(coef: np.ndarray, feature_names: List[str], out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(coef))
    ax.bar(x_pos, coef, color=["steelblue" if c >= 0 else "tomato" for c in coef])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Ridge coefficient")
    ax.set_title("Ridge Regression — Feature Importance (baseline)")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(out_dir, "feature_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Ridge baseline
# ---------------------------------------------------------------------------

def _train_baseline(train_data, val_data, test_data, stats, feature_names):
    """Train a Ridge regression baseline on per-graph summary features."""
    Ridge, StandardScaler, r2_score = _import_sklearn()
    if Ridge is None:
        print("[INFO] scikit-learn not available — skipping Ridge baseline.")
        return None, None, None

    X_tr = _summary_features(train_data)
    X_va = _summary_features(val_data)
    X_te = _summary_features(test_data)

    # Targets in original scale
    y_tr = np.array([d.y.item() * stats["target_std"] + stats["target_mean"] for d in train_data])
    y_va = np.array([d.y.item() * stats["target_std"] + stats["target_mean"] for d in val_data])
    y_te = np.array([d.y.item() * stats["target_std"] + stats["target_mean"] for d in test_data])

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_s, y_tr)

    val_preds = ridge.predict(X_va_s)
    test_preds = ridge.predict(X_te_s)

    val_met = _metrics(val_preds, y_va)
    test_met = _metrics(test_preds, y_te)

    print(f"\n  Ridge Baseline (Val)  — R²={val_met['r2']:.4f}  MAE={val_met['mae']:.4f}  RMSE={val_met['rmse']:.4f}")
    print(f"  Ridge Baseline (Test) — R²={test_met['r2']:.4f}  MAE={test_met['mae']:.4f}  RMSE={test_met['rmse']:.4f}")

    return ridge, scaler, (test_preds, y_te, test_met, ridge.coef_)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    csv_path: str,
    cache_dir: str,
    out_dir: str,
    epochs: int = 150,
    batch_size: int = 32,
    hidden: int = 128,
    n_conv_layers: int = 3,
    lr: float = 3e-4,
    dropout: float = 0.2,
    max_samples: int | None = None,
    seed: int = 42,
    force_regen: bool = False,
) -> None:
    import torch
    from torch_geometric.data import DataLoader

    _set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  PoreNetGNN Training")
    print(f"  Device   : {device}")
    print(f"  CSV      : {csv_path}")
    print(f"  Cache    : {cache_dir}")
    print(f"  Output   : {out_dir}")
    print(f"  Epochs   : {epochs}  |  Batch : {batch_size}  |  LR : {lr}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Build / load dataset
    # ------------------------------------------------------------------
    # Add repo root to path so data_pipeline can import NETWORK_GENERATION_FINAL
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(here, "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from CODE.gnn.data_pipeline import build_dataset, dataset_stats, normalize_dataset

    print("--- Step 1: Loading dataset ---")
    raw_dataset = build_dataset(
        csv_path=csv_path,
        cache_dir=cache_dir,
        max_samples=max_samples,
        force_regen=force_regen,
        verbose=True,
    )
    if len(raw_dataset) < 10:
        raise ValueError(
            f"Dataset too small ({len(raw_dataset)} graphs). "
            "Increase max_samples or provide a richer CSV."
        )

    # ------------------------------------------------------------------
    # 2. Split
    # ------------------------------------------------------------------
    print("\n--- Step 2: Splitting dataset ---")
    train_raw, val_raw, test_raw = _train_val_test_split(raw_dataset, seed=seed)
    print(f"  Train={len(train_raw)}  Val={len(val_raw)}  Test={len(test_raw)}")

    # ------------------------------------------------------------------
    # 3. Normalise (fit on train only)
    # ------------------------------------------------------------------
    print("\n--- Step 3: Normalising features ---")
    stats = dataset_stats(train_raw)
    print(f"  Node features   : {stats['n_node_features']}")
    print(f"  Edge features   : {stats['n_edge_features']}")
    print(f"  Target mean/std : {stats['target_mean']:.3f} / {stats['target_std']:.3f}")

    train_data = normalize_dataset(train_raw, stats)
    val_data = normalize_dataset(val_raw, stats)
    test_data = normalize_dataset(test_raw, stats)

    # ------------------------------------------------------------------
    # 4. Build model
    # ------------------------------------------------------------------
    print("\n--- Step 4: Building model ---")
    from CODE.gnn.model import build_model
    model = build_model(stats, hidden=hidden, n_conv_layers=n_conv_layers, dropout=dropout)
    model = model.to(device)
    print(f"  Trainable parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    print("\n--- Step 5: Training ---")
    best_val_loss = float("inf")
    best_model_path = os.path.join(out_dir, "best_model.pt")
    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss = _run_epoch(model, train_loader, optimizer, device, train=True)
        va_loss = _run_epoch(model, val_loader, optimizer, device, train=False)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), best_model_path)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Ep {epoch:4d}/{epochs}  "
                  f"train={tr_loss:.5f}  val={va_loss:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"[{elapsed:.0f}s]")

    _plot_training_curves(train_losses, val_losses, out_dir)

    # ------------------------------------------------------------------
    # 6. Test evaluation (load best weights)
    # ------------------------------------------------------------------
    print("\n--- Step 6: Test evaluation ---")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    gnn_preds, gnn_targets = _evaluate(model, test_data, stats, device, batch_size)
    gnn_metrics = _metrics(gnn_preds, gnn_targets)
    print(f"\n  GNN (Test) — R²={gnn_metrics['r2']:.4f}  "
          f"MAE={gnn_metrics['mae']:.4f}  RMSE={gnn_metrics['rmse']:.4f}")

    # ------------------------------------------------------------------
    # 7. Ridge baseline
    # ------------------------------------------------------------------
    print("\n--- Step 7: Ridge regression baseline ---")
    feature_names = [
        "node_log_diam_mean", "node_log_vol_mean",
        "node_x_mean", "node_y_mean", "node_z_mean",
        "edge_log_d_mean", "edge_log_l_mean", "edge_log_g_mean",
        "graph_log_L", "graph_log_rho",
        "log_num_nodes", "log_num_edges",
    ]
    _, _, baseline_result = _train_baseline(train_data, val_data, test_data, stats, feature_names)
    baseline_metrics = None
    if baseline_result is not None:
        _, _, baseline_metrics, coef = baseline_result
        _plot_baseline_coefficients(coef, feature_names, out_dir)

    # ------------------------------------------------------------------
    # 8. Plots
    # ------------------------------------------------------------------
    print("\n--- Step 8: Generating plots ---")
    _plot_parity(gnn_preds, gnn_targets, gnn_metrics, baseline_metrics, out_dir)
    _plot_residuals(gnn_preds, gnn_targets, out_dir)

    # ------------------------------------------------------------------
    # 9. Save results summary
    # ------------------------------------------------------------------
    summary = {
        "gnn": gnn_metrics,
        "baseline_ridge": baseline_metrics,
        "n_train": len(train_data),
        "n_val": len(val_data),
        "n_test": len(test_data),
        "n_params": model.count_parameters(),
        "best_val_loss": float(best_val_loss),
        "epochs": epochs,
        "hidden": hidden,
        "n_conv_layers": n_conv_layers,
        "lr": lr,
        "target_mean": stats["target_mean"],
        "target_std": stats["target_std"],
    }
    import json
    summary_path = os.path.join(out_dir, "results_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved results summary: {summary_path}")

    print(f"\n{'='*60}")
    print("  Training complete!")
    print(f"  Best model   : {best_model_path}")
    print(f"  GNN  R²      : {gnn_metrics['r2']:.4f}")
    if baseline_metrics:
        print(f"  Ridge R²     : {baseline_metrics['r2']:.4f}")
        improvement = gnn_metrics["r2"] - baseline_metrics["r2"]
        print(f"  ΔR² (GNN−Ridge): {improvement:+.4f}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train PoreNetGNN on pore-network transport data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to sweep results CSV (e.g. CODE/FINAL SET/const_density_1e12/sweep_const_density_1e12_full.csv)"
    )
    parser.add_argument(
        "--cache", required=True,
        help="Directory to cache processed .pt graph files"
    )
    parser.add_argument(
        "--out", default="CODE/gnn/results",
        help="Output directory for plots and model weights"
    )
    parser.add_argument("--epochs",      type=int,   default=150)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--hidden",      type=int,   default=128)
    parser.add_argument("--conv-layers", type=int,   default=3,   dest="n_conv_layers")
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--max-samples", type=int,   default=None)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--force-regen", action="store_true",
                        help="Force regeneration of cached graphs")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        csv_path=args.csv,
        cache_dir=args.cache,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden=args.hidden,
        n_conv_layers=args.n_conv_layers,
        lr=args.lr,
        dropout=args.dropout,
        max_samples=args.max_samples,
        seed=args.seed,
        force_regen=args.force_regen,
    )
