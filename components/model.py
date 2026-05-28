import torch
import torch.nn as nn
from torch_geometric.nn import (
    GraphNorm, 
    MessagePassing,
    global_add_pool,
    JumpingKnowledge
    )
from torch_geometric.utils import (
    degree, 
    get_laplacian, 
    to_dense_adj
    )
try:
    from components.attention import (
        BondFastAttention,
        MultiAtomAttention,
        SublayerConnection,
    )
    from train.loss import SupervisedUncertainty
    from components.activation import get_activation
except ImportError:
    from attention import (
        BondFastAttention,
        MultiAtomAttention,
        SublayerConnection,
    )
    from loss import SupervisedUncertainty
    from activation import get_activation


def resolve_heads(hidden_dim, requested_heads):
    hidden_dim = int(max(1, hidden_dim))
    requested_heads = int(max(1, requested_heads))
    return min(hidden_dim, requested_heads)


def validate_transformer_dims(
    agg_hidden_dims,
    edge_heads,
    node_heads,
):
    invalid_edge_dims = [
        dim for dim in agg_hidden_dims
        if int(dim) % int(edge_heads) != 0
    ]
    if invalid_edge_dims:
        raise ValueError(
            "All message-passing hidden dimensions must be divisible "
            f"by edge_heads={edge_heads}. Invalid values: "
            f"{invalid_edge_dims}"
        )

    node_rep_dim = int(sum(agg_hidden_dims))
    if node_rep_dim % int(node_heads) != 0:
        raise ValueError(
            "The concatenated node representation dimension "
            f"(sum(agg_hidden_dims)={node_rep_dim}) must be divisible "
            f"by node_heads={node_heads}."
        )



class EdgeFormerBlock(MessagePassing):
    def __init__(
        self,
        input_dim,
        output_dim,
        edge_dim,
        dropout_rate,
        heads):

        super().__init__(aggr="add")
        self.heads = resolve_heads(output_dim, heads)
        self.edge_proj_dim = edge_dim
        if edge_dim % self.heads != 0:
            self.edge_proj_dim = self.heads * (
                edge_dim // self.heads + 1
            )
            self.edge_proj_adjust = nn.Linear(
                edge_dim, self.edge_proj_dim
            )
        else:
            self.edge_proj_adjust = None

        self.input_adjust = nn.Linear(input_dim, output_dim)
        self.node_proj = nn.Linear(output_dim, output_dim)
        self.edge_proj = nn.Linear(self.edge_proj_dim, output_dim)
        self.bond_attn = BondFastAttention(
            hidden_size=output_dim,
            heads=self.heads,
            dropout=dropout_rate,
        )
        self.bond_residual = SublayerConnection(dropout=dropout_rate)
        self.residual = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )
        self.layer_norm = GraphNorm(output_dim)
        self.att_dropout = nn.Dropout(dropout_rate)

    def reset_parameters(self):
        if self.edge_proj_adjust is not None:
            self.edge_proj_adjust.reset_parameters()
        self.input_adjust.reset_parameters()
        self.node_proj.reset_parameters()
        self.edge_proj.reset_parameters()
        self.bond_attn.reset_parameters()
        if hasattr(self.residual, "reset_parameters"):
            self.residual.reset_parameters()
        self.layer_norm.reset_parameters()

    @staticmethod
    def _construct_edge_scopes(edge_batch):
        if edge_batch.numel() == 0:
            return [], None
        sorted_batch, idx = torch.sort(edge_batch)
        counts = torch.bincount(sorted_batch)
        scopes = []
        start = 0
        for count in counts.tolist():
            if count <= 0:
                continue
            scopes.append((start, int(count)))
            start += int(count)
        return scopes, idx

    def message(self, x_j, edge_attr, edge_batch):
        node_messages = self.input_adjust(x_j)
        if self.edge_proj_adjust is not None:
            edge_attr = self.edge_proj_adjust(edge_attr)
        bond_messages = self.edge_proj(edge_attr)

        scopes, sort_idx = self._construct_edge_scopes(edge_batch)
        if sort_idx is not None:
            bond_messages = bond_messages[sort_idx]
            node_messages = node_messages[sort_idx]

        att_out = self.bond_attn(bond_messages, scopes)
        out_sorted = self.bond_residual(
            self.node_proj(node_messages),
            att_out,
        ).view(-1, att_out.size(-1))

        if sort_idx is None:
            return out_sorted

        restore_idx = torch.empty_like(sort_idx)
        restore_idx[sort_idx] = torch.arange(
            sort_idx.numel(),
            device=sort_idx.device,
        )
        return out_sorted[restore_idx]

    def forward(self, x, edge_index, edge_attr, batch):
        x_in = x
        row = edge_index[0]
        deg = degree(row, x.size(0), dtype=x.dtype).to(x.device)
        inv_deg = deg.pow(-0.5)
        inv_deg[torch.isinf(inv_deg)] = 0
        x_scaled = x * inv_deg.unsqueeze(-1)
        edge_batch = batch.index_select(0, row)
        out = self.propagate(
            edge_index=edge_index,
            x=x_scaled,
            edge_attr=edge_attr,
            edge_batch=edge_batch,
        )
        out = out + self.residual(x_in)
        out = self.layer_norm(out, batch)
        out = self.att_dropout(out)
        return out


class NodeReadoutBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        dropout_rate,
        heads,
        activation):

        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.heads = resolve_heads(hidden_dim, heads)
        self.attn = MultiAtomAttention(
            node_feat_dim=self.hidden_dim,
            dropout=dropout_rate,
            activation=activation,
            device=None,
            heads=self.heads,
        )
        self.norm = GraphNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def reset_parameters(self):
        self.attn.reset_parameters()
        self.norm.reset_parameters()

    @staticmethod
    def _block_diag_from_dense(dense_batch, counts):
        blocks = []
        for graph_idx, n_nodes in enumerate(counts.tolist()):
            if n_nodes <= 0:
                continue
            blocks.append(dense_batch[graph_idx, :n_nodes, :n_nodes])
        if not blocks:
            return dense_batch.new_zeros((0, 0))
        return torch.block_diag(*blocks)

    def forward(self, x, edge_index, batch):
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        counts = torch.bincount(batch, minlength=num_graphs)
        lap_idx, lap_weight = get_laplacian(
            edge_index,
            edge_weight=torch.ones(
                edge_index.size(1),
                device=x.device,
                dtype=x.dtype,
                ),
            normalization="sym",
        )
        lap_dense = to_dense_adj(
            lap_idx,
            batch=batch,
            edge_attr=lap_weight,
        )
        adj_dense = to_dense_adj(
            edge_index,
            batch=batch,
        ).to(x.dtype)
        laplacian = self._block_diag_from_dense(lap_dense, counts)
        adjacency = self._block_diag_from_dense(adj_dense, counts)
        att_out, _ = self.attn(
            x,
            edge_index,
            batch,
            laplacian,
            adjacency,
        )
        out = self.norm(x + att_out, batch)
        out = self.dropout(out)
        return out


class MPNNLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        edge_dim,
        dropout_rate,
        edge_heads,
    ):

        super().__init__()
        self.edge_layer = EdgeFormerBlock(
            input_dim=input_dim,
            output_dim=output_dim,
            edge_dim=edge_dim,
            dropout_rate=dropout_rate,
            heads=edge_heads,
        )

    def reset_parameters(self):
        self.edge_layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.edge_layer(x, edge_index, edge_attr, batch)
        return x




class network(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        agg_hidden_dims,
        num_agg_layers,
        lin_hidden_dims,
        num_lin_layers,
        activation,
        dropout_rate,
        num_tasks,
        mc_class_counts=None,
        edge_heads=4,
        node_heads=4):

        super(network, self).__init__()
        validate_transformer_dims(
            agg_hidden_dims,
            edge_heads,
            node_heads,
        )

        self.uncertainty = SupervisedUncertainty(
            num_tasks=num_tasks)

        self.num_tasks = num_tasks
        self.saved_embeddings = []

        if mc_class_counts is None:
            mc_class_counts = torch.zeros(num_tasks, dtype=torch.long)
        elif not torch.is_tensor(mc_class_counts):
            mc_class_counts = torch.tensor(mc_class_counts, dtype=torch.long)
        else:
            mc_class_counts = mc_class_counts.to(torch.long)

        if mc_class_counts.numel() != num_tasks:
            raise ValueError("mc_class_counts must have length num_tasks")

        self.register_buffer("mc_class_counts", mc_class_counts)
        self.mc_task_indices = [
            i for i in range(num_tasks)
            if int(self.mc_class_counts[i].item()) > 1
            ]

        self.agg_layers = nn.ModuleList()
        dims = [agg_hidden_dims[i]
            for i in range(num_agg_layers)
            ]
        input_dims = [node_dim] + dims[:-1]
        for i in range(num_agg_layers):
            self.agg_layers.append(
                MPNNLayer(
                    input_dims[i],
                    dims[i],
                    edge_dim,
                    dropout_rate,
                    edge_heads=edge_heads,
                )
            )
        self.jk = JumpingKnowledge(mode='cat')
        self.virtualnode_embedding = nn.ParameterList([
            nn.Parameter(torch.zeros(1, d))
            for d in input_dims])
        self.virtualnode_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_d, in_d),
                nn.LayerNorm(in_d),
                nn.ReLU(),
                nn.Linear(in_d, in_d))
            for in_d, out_d in zip(input_dims, dims)])
        self.lin_layers = nn.ModuleList()
        self.node_rep_dim = sum(dims)
        self.node_readout = NodeReadoutBlock(
            hidden_dim=self.node_rep_dim,
            dropout_rate=dropout_rate,
            heads=node_heads,
            activation=activation,
        )
        for i in range(num_lin_layers):
            in_d = self.node_rep_dim if i == 0 else lin_hidden_dims[i-1]
            out_d = lin_hidden_dims[i]
            self.lin_layers.append(
                nn.Sequential(
                    nn.Linear(in_d, out_d),
                    get_activation(activation),
                    nn.Dropout(dropout_rate)))
        self.embedding_dim = lin_hidden_dims[-1]
        self.embedding_layer = nn.Linear(
            lin_hidden_dims[-1],
            self.embedding_dim)

        self.output_layer = nn.Linear(
            self.embedding_dim,
            num_tasks)
        self.mc_heads = nn.ModuleDict({
            str(i): nn.Linear(
                self.embedding_dim,
                int(self.mc_class_counts[i].item()))
            for i in self.mc_task_indices
            })

    def forward(
        self,
        data,
        save_embeddings=False,
        return_penultimate=False,
        return_embeddings=False,
        return_representations=False):

        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch)
        num_graphs = batch.max().item() + 1
        v_nodes = [emb.expand(num_graphs, -1)
                for emb in self.virtualnode_embedding
                ]
        xs = []
        for i, layer in enumerate(self.agg_layers):
            v = v_nodes[i][batch]
            x = x + v
            x = layer(x, edge_index, edge_attr, batch)
            xs.append(x)
            pooled = global_add_pool(x, batch)
            delta = self.virtualnode_mlp[i](pooled)
            v_nodes[i] = v_nodes[i] + delta
        node_embeddings = self.jk(xs)
        node_embeddings = self.node_readout(
            node_embeddings,
            edge_index,
            batch,
        )
        x = global_add_pool(node_embeddings, batch)

        for layer in self.lin_layers:
            x = layer(x)

        embeddings = self.embedding_layer(x)
        penultimate = embeddings.clone()

        out = self.output_layer(embeddings)

        if save_embeddings:
            self.saved_embeddings.append(
                penultimate.detach().cpu())
        if return_penultimate or return_embeddings:
            return penultimate
        if return_representations:
            prediction = out
            if self.mc_heads:
                mc_logits = {}
                for idx, head in self.mc_heads.items():
                    mc_logits[int(idx)] = head(embeddings)
                prediction = {
                    "scalar": out,
                    "mc_logits": mc_logits,
                }
            return {
                "graph_embeddings": embeddings,
                "node_embeddings": node_embeddings,
                "predictions": prediction,
            }

        if not self.mc_heads:
            return out
        mc_logits = {}
        for idx, head in self.mc_heads.items():
            mc_logits[int(idx)] = head(embeddings)

        return {"scalar": out, "mc_logits": mc_logits}
