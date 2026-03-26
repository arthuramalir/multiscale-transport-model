"""data_pipeline.py — Build PyTorch Geometric graph datasets from pore-network CSVs.

Each row in the sweep CSV corresponds to one simulation run:
    seed + L_m  →  rebuild Voronoi network  →  extract node/edge features  →  PyG Data object

Node features (per pore):
    0: log10(pore diameter / 1e-6)      [dimensionless, log-scaled]
    1: log10(pore volume / 1e-18)        [dimensionless, log-scaled]
    2: x / L_m                           [0-1 normalised coordinate]
    3: y / L_m
    4: z / L_m

Edge features (per throat, stored on both directed edges):
    0: log10(throat diameter / 1e-6)
    1: log10(throat length / 1e-6)
    2: log10(diffusive conductance / 1e-20)

Graph-level features:
    [log10(L_m / 1e-6),  log10(density)]   (shape: [2])

Target:
    log10(dir_D_eff)   (scalar, regression)

Usage
-----
    from CODE.gnn.data_pipeline import build_dataset, load_dataset

    dataset = build_dataset(
        csv_path="CODE/FINAL SET/const_density_1e12/sweep_const_density_1e12_full.csv",
        cache_dir="CODE/gnn/cache/const_density_1e12",
        max_samples=500,        # set None for all
    )
    # Returns a list of torch_geometric.data.Data objects

Caching
-------
    Processed `.pt` files are written to `cache_dir` so subsequent calls skip
    network regeneration (which requires OpenPNM).  Pass `force_regen=True` to
    rebuild.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import warnings
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Deferred heavy imports so module can be loaded for inspection without them
# ---------------------------------------------------------------------------
_TORCH_AVAILABLE = False
_OPN_AVAILABLE = False


def _require_torch() -> None:
    global _TORCH_AVAILABLE
    if not _TORCH_AVAILABLE:
        try:
            import torch  # noqa: F401
            import torch_geometric  # noqa: F401
            _TORCH_AVAILABLE = True
        except ImportError as e:
            raise ImportError(
                "PyTorch and PyTorch Geometric are required for the GNN pipeline.\n"
                "Install them with:\n"
                "  pip install torch torch_geometric\n"
                f"Original error: {e}"
            ) from e


def _require_openpnm() -> None:
    global _OPN_AVAILABLE
    if not _OPN_AVAILABLE:
        try:
            import openpnm  # noqa: F401
            _OPN_AVAILABLE = True
        except ImportError as e:
            raise ImportError(
                "OpenPNM is required to regenerate pore networks.\n"
                "Install it with:  pip install openpnm\n"
                f"Original error: {e}"
            ) from e


# ---------------------------------------------------------------------------
# Internal: network regeneration and feature extraction
# ---------------------------------------------------------------------------

def _safe_log10(x: np.ndarray, floor: float = 1e-40) -> np.ndarray:
    """log10 with a floor to avoid -inf."""
    return np.log10(np.maximum(x, floor))


def _build_network(seed: int, L: float, density: float):
    """Regenerate Voronoi pore network from (seed, L, density).

    Adds the `CODE/FINAL SET` directory to sys.path so
    NETWORK_GENERATION_FINAL can be imported regardless of working directory.
    """
    _require_openpnm()
    # Locate NETWORK_GENERATION_FINAL relative to this file
    gnn_dir = os.path.dirname(os.path.abspath(__file__))
    final_set_dir = os.path.normpath(os.path.join(gnn_dir, "..", "FINAL SET"))
    code_dir = os.path.normpath(os.path.join(gnn_dir, ".."))
    for d in (final_set_dir, code_dir):
        if d not in sys.path:
            sys.path.insert(0, d)

    from NETWORK_GENERATION_FINAL import build_voronoi_network  # type: ignore

    domain = (L, L, L)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pn = build_voronoi_network(
            domain_size=domain,
            pore_density=density,
            pore_size_dist={"name": "lognorm", "s": 0.25, "scale": 20e-6},
            throat_size_dist={"name": "weibull_min", "c": 2.5, "scale": 8e-6},
            correlate_pore_throat=False,
            seed=seed,
            save_files=False,
        )
    return pn


def _compute_conductance(pn, air) -> np.ndarray:
    """Compute throat diffusive conductance (D * A / L) for all throats."""
    import openpnm as op  # noqa: F401

    try:
        pore_diff = air["pore.diffusivity"]
    except KeyError:
        pore_diff = np.full(pn.Np, 1e-5)

    conns = pn["throat.conns"]
    td = pn["throat.diameter"].copy()
    tl = pn["throat.length"].copy()

    # Repair invalid values
    bad_td = ~np.isfinite(td) | (td <= 0)
    if np.any(bad_td):
        td[bad_td] = np.nanmean(td[td > 0]) if np.any(td > 0) else 1e-9
    bad_tl = ~np.isfinite(tl) | (tl <= 0)
    if np.any(bad_tl):
        tl[bad_tl] = np.nanmean(tl[tl > 0]) if np.any(tl > 0) else 1e-9

    D_mean = 0.5 * (pore_diff[conns[:, 0]] + pore_diff[conns[:, 1]])
    D_mean = np.where(np.isfinite(D_mean), D_mean, np.nanmean(D_mean))
    area = np.pi * (td / 2.0) ** 2
    g = D_mean * area / tl
    g = np.where(np.isfinite(g) & (g > 0), g, 1e-40)
    return g


def _extract_graph(pn, air, L: float, density: float, target_log_deff: float):
    """Extract node features, edge index, edge features, and target from a network.

    Returns a dict with keys matching torch_geometric.data.Data fields.
    """
    import torch  # noqa: F401

    coords = pn["pore.coords"]  # (Np, 3)
    pore_d = pn["pore.diameter"]  # (Np,)
    try:
        pore_v = pn["pore.volume"]
    except KeyError:
        # Approximate sphere volume if unavailable
        pore_v = (4.0 / 3.0) * np.pi * (pore_d / 2.0) ** 3

    # --- Node features: (Np, 5) ---
    node_x = np.column_stack([
        _safe_log10(pore_d / 1e-6),          # 0: log10 diameter in µm
        _safe_log10(pore_v / 1e-18),          # 1: log10 volume in µm³
        coords[:, 0] / L,                     # 2: x/L
        coords[:, 1] / L,                     # 3: y/L
        coords[:, 2] / L,                     # 4: z/L
    ]).astype(np.float32)                     # (Np, 5)

    # --- Edge index + edge features: undirected → two directed copies ---
    conns = pn["throat.conns"]  # (Nt, 2)
    throat_d = pn["throat.diameter"].copy()
    throat_l = pn["throat.length"].copy()
    bad_td = ~np.isfinite(throat_d) | (throat_d <= 0)
    bad_tl = ~np.isfinite(throat_l) | (throat_l <= 0)
    if np.any(bad_td):
        throat_d[bad_td] = np.nanmean(throat_d[throat_d > 0]) if np.any(throat_d > 0) else 1e-9
    if np.any(bad_tl):
        throat_l[bad_tl] = np.nanmean(throat_l[throat_l > 0]) if np.any(throat_l > 0) else 1e-9

    cond = _compute_conductance(pn, air)

    edge_attr_fwd = np.column_stack([
        _safe_log10(throat_d / 1e-6),     # 0: log10 diameter in µm
        _safe_log10(throat_l / 1e-6),     # 1: log10 length in µm
        _safe_log10(cond / 1e-20),        # 2: log10 conductance / 1e-20
    ]).astype(np.float32)                 # (Nt, 3)

    # Bidirectional: forward + reverse edges
    src = np.concatenate([conns[:, 0], conns[:, 1]])
    dst = np.concatenate([conns[:, 1], conns[:, 0]])
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)   # (2, 2*Nt)
    edge_attr = np.concatenate([edge_attr_fwd, edge_attr_fwd], axis=0)  # (2*Nt, 3)

    # --- Graph-level feature ---
    graph_feat = np.array([
        math.log10(max(L / 1e-6, 1e-30)),
        math.log10(max(density, 1.0)),
    ], dtype=np.float32)

    return {
        "x": node_x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "graph_feat": graph_feat,
        "y": np.float32(target_log_deff),
        "num_nodes": pn.Np,
    }


def _to_pyg_data(graph_dict: dict):
    """Convert the raw numpy dict to a torch_geometric.data.Data object."""
    import torch
    from torch_geometric.data import Data

    return Data(
        x=torch.from_numpy(graph_dict["x"]),
        edge_index=torch.from_numpy(graph_dict["edge_index"]),
        edge_attr=torch.from_numpy(graph_dict["edge_attr"]),
        graph_feat=torch.from_numpy(graph_dict["graph_feat"]).unsqueeze(0),  # (1, 2)
        y=torch.tensor([graph_dict["y"]], dtype=torch.float32),
        num_nodes=graph_dict["num_nodes"],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataset(
    csv_path: str,
    cache_dir: str,
    max_samples: Optional[int] = None,
    force_regen: bool = False,
    skip_errors: bool = True,
    verbose: bool = True,
) -> List:
    """Build a list of PyG Data objects from a sweep CSV.

    Parameters
    ----------
    csv_path     : Path to the sweep results CSV (e.g. sweep_const_density_1e12_full.csv)
    cache_dir    : Directory where processed `.pt` graph files are cached
    max_samples  : Cap the number of samples processed (None = all)
    force_regen  : Re-build even if cached `.pt` exists
    skip_errors  : If True, skip rows that fail; otherwise raise
    verbose      : Print progress messages

    Returns
    -------
    List of torch_geometric.data.Data objects
    """
    _require_torch()
    import torch

    os.makedirs(cache_dir, exist_ok=True)

    # Load CSV
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    if max_samples is not None:
        rows = rows[:max_samples]

    dataset = []
    n_cached = 0
    n_built = 0
    n_skipped = 0

    for i, row in enumerate(rows):
        # Skip rows with errors
        if row.get("error", "").strip():
            n_skipped += 1
            continue

        seed = int(row["seed"])
        L = float(row["L_m"])
        density = float(row.get("density", 1e12))

        raw_deff = row.get("dir_D_eff", "")
        if not raw_deff or raw_deff.strip() == "":
            n_skipped += 1
            continue
        dir_D_eff = float(raw_deff)
        if not np.isfinite(dir_D_eff) or dir_D_eff <= 0:
            n_skipped += 1
            continue

        target_log_deff = math.log10(dir_D_eff)

        # Cache file name encodes the key inputs for reproducibility
        cache_fname = f"seed{seed}_L{L:.6g}_rho{density:.2e}.pt"
        cache_path = os.path.join(cache_dir, cache_fname)

        if os.path.exists(cache_path) and not force_regen:
            data = torch.load(cache_path, weights_only=False)
            # Update target in case CSV was edited
            data.y = torch.tensor([target_log_deff], dtype=torch.float32)
            dataset.append(data)
            n_cached += 1
        else:
            try:
                pn = _build_network(seed, L, density)
                # Retrieve the Air phase (created inside build_voronoi_network)
                try:
                    air = pn.project.phases()["Air"]
                except Exception:
                    import openpnm as op
                    air = op.phase.Phase(network=pn)
                    air["pore.diffusivity"] = np.full(pn.Np, 1e-5)

                graph_dict = _extract_graph(pn, air, L, density, target_log_deff)
                data = _to_pyg_data(graph_dict)
                torch.save(data, cache_path)
                dataset.append(data)
                n_built += 1
            except Exception as exc:
                if skip_errors:
                    if verbose:
                        print(f"  [WARN] skipping seed={seed} L={L:.2e}: {exc}")
                    n_skipped += 1
                    continue
                else:
                    raise

        if verbose and (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(rows)} rows "
                  f"(cached={n_cached}, built={n_built}, skipped={n_skipped})")

    if verbose:
        print(f"\nDataset ready: {len(dataset)} graphs "
              f"(from cache={n_cached}, freshly built={n_built}, skipped={n_skipped})")
    return dataset


def load_dataset(cache_dir: str) -> List:
    """Load all cached `.pt` graphs from a directory (no OpenPNM required)."""
    _require_torch()
    import torch

    pt_files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pt"))
    if not pt_files:
        raise FileNotFoundError(
            f"No cached .pt files found in '{cache_dir}'.\n"
            "Run build_dataset() first to generate the cache."
        )
    dataset = [torch.load(os.path.join(cache_dir, f), weights_only=False) for f in pt_files]
    print(f"Loaded {len(dataset)} cached graphs from '{cache_dir}'")
    return dataset


def dataset_stats(dataset: list) -> dict:
    """Compute mean/std for all node features, edge features, and targets."""
    import torch

    all_x = torch.cat([d.x for d in dataset], dim=0)
    all_e = torch.cat([d.edge_attr for d in dataset], dim=0)
    all_y = torch.cat([d.y for d in dataset], dim=0)

    return {
        "node_mean": all_x.mean(0),
        "node_std": all_x.std(0).clamp(min=1e-8),
        "edge_mean": all_e.mean(0),
        "edge_std": all_e.std(0).clamp(min=1e-8),
        "target_mean": all_y.mean().item(),
        "target_std": all_y.std().item(),
        "n_graphs": len(dataset),
        "n_node_features": all_x.shape[1],
        "n_edge_features": all_e.shape[1],
    }


def normalize_dataset(dataset: list, stats: dict) -> list:
    """Return a new dataset with normalised node/edge features and target."""
    import torch
    from torch_geometric.data import Data

    normed = []
    for d in dataset:
        x_n = (d.x - stats["node_mean"]) / stats["node_std"]
        e_n = (d.edge_attr - stats["edge_mean"]) / stats["edge_std"]
        y_n = (d.y - stats["target_mean"]) / stats["target_std"]
        normed.append(Data(
            x=x_n,
            edge_index=d.edge_index.clone(),
            edge_attr=e_n,
            graph_feat=d.graph_feat.clone() if hasattr(d, "graph_feat") else None,
            y=y_n,
            num_nodes=d.num_nodes,
        ))
    return normed


# ---------------------------------------------------------------------------
# CLI helper: python -m CODE.gnn.data_pipeline --csv <path> --cache <dir>
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build and cache PyG graph dataset from a sweep CSV."
    )
    parser.add_argument("--csv", required=True, help="Path to sweep results CSV")
    parser.add_argument("--cache", required=True, help="Cache directory for .pt files")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Force regeneration")
    args = parser.parse_args()

    ds = build_dataset(
        csv_path=args.csv,
        cache_dir=args.cache,
        max_samples=args.max_samples,
        force_regen=args.force,
        verbose=True,
    )
    stats = dataset_stats(ds)
    print("\nDataset statistics:")
    print(f"  Graphs            : {stats['n_graphs']}")
    print(f"  Node features     : {stats['n_node_features']}")
    print(f"  Edge features     : {stats['n_edge_features']}")
    print(f"  Target log10(Deff): mean={stats['target_mean']:.3f}, std={stats['target_std']:.3f}")
