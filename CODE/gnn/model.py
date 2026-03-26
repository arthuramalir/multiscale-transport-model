"""model.py — Graph Neural Network for pore-network transport prediction.

Architecture: GraphSAGE encoder → global pooling → MLP decoder

    Input:  pore network graph (variable-size nodes + edges)
    Output: log10(D_eff)  [scalar, graph-level regression]

The design intentionally mirrors the physical intuition:
  • GraphSAGE message-passing aggregates local neighbourhood geometry
    (pore sizes, throat conductances) over multiple hops.
  • Global mean+max pooling produces a fixed-size graph embedding that
    captures both average behaviour and worst-case bottlenecks.
  • Optional graph-level features (domain size, pore density) are
    concatenated before the MLP regression head.

Usage
-----
    from CODE.gnn.model import PoreNetGNN

    model = PoreNetGNN(
        node_in=5,   # number of node features
        edge_in=3,   # number of edge features
        graph_feat_in=2,  # number of graph-level features
    )
    out = model(data)   # data: torch_geometric.data.Data
    # out.shape == (batch_size, 1)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm, global_mean_pool, global_max_pool


class EdgeEncoder(nn.Module):
    """Project raw edge features to a hidden representation used as
    virtual node features by concatenating them to the source-node embedding
    before SAGEConv aggregation.

    This is a lightweight way to inject edge information into a SAGEConv-based
    model without switching to a fully edge-aware convolution.
    """

    def __init__(self, edge_in: int, hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.mlp(edge_attr)


class PoreNetGNN(nn.Module):
    """GraphSAGE-based GNN for predicting log10(D_eff) of a pore network.

    Parameters
    ----------
    node_in        : Number of input node features (default 5).
    edge_in        : Number of input edge features (default 3).
    graph_feat_in  : Number of graph-level context features (default 2;
                     set 0 to disable).
    hidden         : Width of all hidden layers (default 128).
    n_conv_layers  : Number of SAGEConv message-passing rounds (default 3).
    dropout        : Dropout probability in the MLP decoder (default 0.2).
    """

    def __init__(
        self,
        node_in: int = 5,
        edge_in: int = 3,
        graph_feat_in: int = 2,
        hidden: int = 128,
        n_conv_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.graph_feat_in = graph_feat_in

        # --- Edge encoder ---
        self.edge_encoder = EdgeEncoder(edge_in, hidden)

        # --- Node input projection ---
        # Concatenate node features with aggregated edge features → richer input
        self.input_proj = nn.Linear(node_in + hidden, hidden)

        # --- Message-passing layers ---
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_conv_layers):
            self.convs.append(SAGEConv(hidden, hidden))
            self.norms.append(BatchNorm(hidden))

        # --- Global pooling → graph embedding ---
        # mean pool + max pool concatenated → 2 * hidden
        pool_out = 2 * hidden

        # --- Optional graph-level feature fusion ---
        if graph_feat_in > 0:
            self.graph_feat_proj = nn.Linear(graph_feat_in, hidden)
            mlp_in = pool_out + hidden
        else:
            self.graph_feat_proj = None
            mlp_in = pool_out

        # --- Regression MLP head ---
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    # ------------------------------------------------------------------
    def _aggregate_edge_to_node(
        self,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Mean-pool encoded edge features into destination nodes.

        For each node i, returns the mean of all encoded edge features whose
        destination is i. Nodes with no incoming edges get a zero vector.
        """
        enc = self.edge_encoder(edge_attr)          # (E, hidden)
        dst = edge_index[1]                          # destination node indices
        agg = torch.zeros(num_nodes, enc.shape[1], device=enc.device)
        count = torch.zeros(num_nodes, 1, device=enc.device)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(enc), enc)
        count.scatter_add_(0, dst.unsqueeze(1), torch.ones(dst.shape[0], 1, device=enc.device))
        count = count.clamp(min=1.0)
        return agg / count                          # (Np, hidden)

    # ------------------------------------------------------------------
    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data or torch_geometric.data.Batch
            Expected attributes: x, edge_index, edge_attr,
                                 batch (auto-set by DataLoader),
                                 graph_feat (optional).

        Returns
        -------
        Tensor of shape (batch_size, 1) with predicted log10(D_eff).
        """
        x = data.x                      # (N, node_in)
        edge_index = data.edge_index    # (2, E)
        edge_attr = data.edge_attr      # (E, edge_in)
        batch = data.batch              # (N,)  — node-to-graph assignment

        num_nodes = x.size(0)

        # 1. Aggregate edge features to nodes
        edge_agg = self._aggregate_edge_to_node(edge_attr, edge_index, num_nodes)  # (N, hidden)

        # 2. Project node + edge-aggregated features to hidden space
        h = F.relu(self.input_proj(torch.cat([x, edge_agg], dim=-1)))  # (N, hidden)

        # 3. Graph convolution with skip connections
        for conv, norm in zip(self.convs, self.norms):
            h_new = F.relu(norm(conv(h, edge_index)))
            h = h + h_new               # residual (same width → no projection needed)

        # 4. Global pooling: mean + max → (graphs, 2*hidden)
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        graph_emb = torch.cat([h_mean, h_max], dim=-1)  # (B, 2*hidden)

        # 5. Fuse optional graph-level features (domain size, pore density)
        if self.graph_feat_proj is not None and hasattr(data, "graph_feat") and data.graph_feat is not None:
            gf = data.graph_feat                        # (B, 1, graph_feat_in) or (B, graph_feat_in)
            gf = gf.view(graph_emb.size(0), -1)        # (B, graph_feat_in)
            gf_enc = F.relu(self.graph_feat_proj(gf))  # (B, hidden)
            graph_emb = torch.cat([graph_emb, gf_enc], dim=-1)  # (B, 2*hidden + hidden)

        # 6. Regression head
        out = self.mlp(graph_emb)   # (B, 1)
        return out

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Convenience factory with validated defaults
# ---------------------------------------------------------------------------

def build_model(stats: dict, hidden: int = 128, n_conv_layers: int = 3, dropout: float = 0.2) -> PoreNetGNN:
    """Build a PoreNetGNN whose input sizes are inferred from dataset stats.

    Parameters
    ----------
    stats : dict returned by data_pipeline.dataset_stats()
    """
    return PoreNetGNN(
        node_in=stats["n_node_features"],
        edge_in=stats["n_edge_features"],
        graph_feat_in=2,
        hidden=hidden,
        n_conv_layers=n_conv_layers,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch
    from torch_geometric.data import Data, Batch

    torch.manual_seed(0)
    # Tiny synthetic graph: 10 nodes, 20 directed edges, batch size 2
    def _fake_graph():
        Np, Nt = 10, 20
        x = torch.randn(Np, 5)
        src = torch.randint(0, Np, (Nt,))
        dst = torch.randint(0, Np, (Nt,))
        edge_index = torch.stack([src, dst])
        edge_attr = torch.randn(Nt, 3)
        graph_feat = torch.randn(1, 2)
        y = torch.tensor([-9.5])
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    graph_feat=graph_feat, y=y, num_nodes=Np)

    batch = Batch.from_data_list([_fake_graph(), _fake_graph()])
    model = PoreNetGNN()
    out = model(batch)
    print(f"Model output shape : {out.shape}")           # (2, 1)
    print(f"Trainable params   : {model.count_parameters():,}")
    print("Self-test passed ✓")
