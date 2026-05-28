import torch
import torch.nn as nn
from torch_geometric.utils import (
    to_dense_adj, 
    degree
    )


class RWPEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_steps = 10

    def reset_parameters(self):
        return
        
    def forward(
        self,
        edge_index,
        num_nodes):

        dev = edge_index.device
        adj = to_dense_adj(
            edge_index,
            max_num_nodes=num_nodes
            )[0].to(dev)
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1e-10)
        P = adj / deg
        probs = []
        Pk = P
        for _ in range(self.num_steps):
            probs.append(Pk)
            Pk = torch.matmul(Pk, P)
        rw_matrix = torch.stack(probs, dim=-1)
        rw_diag = rw_matrix[
            torch.arange(num_nodes),
            torch.arange(num_nodes), :]
        rw_encoded = torch.nn.functional.normalize(
            rw_diag, p=2, dim=1
            )
        rw_bias = rw_encoded @ rw_encoded.T

        return rw_bias.unsqueeze(0)


class CentralityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_in_deg = 64
        self.max_out_deg = 64
        self.node_feat_dim = 32

        self.in_embed = nn.Parameter(
            torch.randn(
                self.max_in_deg, self.node_feat_dim)
                )
        self.out_embed = nn.Parameter(
            torch.randn(
                self.max_out_deg, self.node_feat_dim)
                )

    def reset_parameters(self):
        nn.init.normal_(self.in_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.out_embed, mean=0.0, std=0.02)
        
    def forward(
        self,
        edge_index_list,
        num_nodes=None):

        if not isinstance(edge_index_list, (list, tuple)):
            edge_index_list = [edge_index_list]
        if isinstance(num_nodes, int):
            num_nodes = [num_nodes]
        if num_nodes is None:
            num_nodes = [None] * len(edge_index_list)
        if len(num_nodes) != len(edge_index_list):
            raise ValueError(
                "num_nodes must match edge_index_list length")

        all_encodings = []
        for edge_index, n_nodes in zip(edge_index_list, num_nodes):
            if n_nodes is None:
                n_nodes = (
                    int(edge_index.max().item()) + 1
                    if edge_index.numel() > 0
                    else 0
                )
            if n_nodes <= 0:
                all_encodings.append(edge_index.new_zeros(
                    (0, self.node_feat_dim)))
                continue
            if edge_index.numel() > 0:
                deg_in = degree(
                    edge_index[1],
                    num_nodes=n_nodes
                    ).long()
                deg_out = degree(
                    edge_index[0],
                    num_nodes=n_nodes
                    ).long()
            else:
                deg_in = torch.zeros(
                    n_nodes, dtype=torch.long,
                    device=edge_index.device)
                deg_out = torch.zeros_like(deg_in)
            deg_in = deg_in.clamp(
                max=self.in_embed.size(0) - 1
                )
            deg_out = deg_out.clamp(
                max=self.out_embed.size(0) - 1
                )
            encoding = (
                self.in_embed[deg_in] + self.out_embed[deg_out]
                )
            all_encodings.append(encoding)

        return torch.cat(all_encodings, dim=0)


class SwiGLU(nn.Module):
    
    def __init__(self, dim, beta=1.0):
        super().__init__()
        self.w = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.beta = beta

    def reset_parameters(self):
        self.w.reset_parameters()
        self.v.reset_parameters()

    def forward(self, x):
        u = self.w(x)
        swish = u * torch.sigmoid(self.beta * u)
        return swish * self.v(x)



def filter_graph(
    laplacian_matrix,
    initial_signal,
    num_filter_steps):

    out = initial_signal
    signal_list = [out.unsqueeze(-1)]
    for _ in range(num_filter_steps - 1):
        out = laplacian_matrix @ out
        signal_list.append(out.unsqueeze(-1))

    return torch.cat(
        signal_list, dim=-1
        )


class PEARLEncoder(nn.Module):
    def __init__(self, phi=nn.Identity(), basis=None):
        super().__init__()
        self.num_filter_steps = 16
        self.num_lin_layers = 2
        self.lin_hidden_dim = 16
        self.lin_out_dim = 16

        self.phi = phi
        self.basis = basis

        if self.num_lin_layers > 0:
            if self.num_lin_layers == 1:
                assert self.lin_hidden_dim == self.lin_out_dim
            self.layers = nn.ModuleList([
                nn.Linear(
                    self.num_filter_steps if i == 0
                        else self.lin_hidden_dim,
                    self.lin_hidden_dim if i < self.num_lin_layers - 1
                        else self.lin_out_dim)
                for i in range(self.num_lin_layers)])
            self.norms = nn.ModuleList([
                nn.BatchNorm1d(
                    self.lin_hidden_dim if i < self.num_lin_layers - 1
                        else self.lin_out_dim)
                for i in range(self.num_lin_layers)])
        self.activation = SwiGLU(
            self.lin_hidden_dim)

    def reset_parameters(self):
        if hasattr(self.phi, "reset_parameters"):
            self.phi.reset_parameters()
        if self.basis is not None and hasattr(
            self.basis, "reset_parameters"
        ):
            self.basis.reset_parameters()
        if hasattr(self, "layers"):
            for layer in self.layers:
                layer.reset_parameters()
        if hasattr(self, "norms"):
            for norm in self.norms:
                norm.reset_parameters()
        self.activation.reset_parameters()

    def forward(
        self,
        laplacian_matrix,
        adjacency_matrix,
        edge_index):

        filtered = filter_graph(
            laplacian_matrix,
            adjacency_matrix,
            self.num_filter_steps
            )
        if self.basis is None:
            out = self.phi(filtered)
        else:
            out = self.phi(filtered, 
            edge_index, self.basis
            )

        return torch.mean(out, dim=-1)

    def out_dims(self):
        return self.phi.out_dims
