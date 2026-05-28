import numpy as np
import torch
import torch.nn.functional as F

def zscore_norm(vals, eps=1e-12):
    vals = np.asarray(vals, dtype=float)
    nz_mask = np.abs(vals) > float(eps)
    if not np.any(nz_mask):
        return vals
    nz = vals[nz_mask]
    mean = nz.mean()
    std = nz.std()
    if std > float(eps):
        vals[nz_mask] = (nz - mean) / (std + 1e-8)
    vals[~nz_mask] = 0.0
    return vals


def _run_attention_forward(model, batch_data):
    prev_mode = model.training
    model.eval()
    with torch.no_grad():
        model(batch_data)
    model.train(prev_mode)


def _resolve_node_attention(model):
    if hasattr(model, "node_readout") and hasattr(model.node_readout, "attn"):
        return model.node_readout.attn
    if hasattr(model, "atom_attention"):
        return model.atom_attention
    return None


def _resolve_edge_layer(model):
    if hasattr(model, "edge_agg_layers") and len(model.edge_agg_layers) > 0:
        return model.edge_agg_layers[-1]
    if hasattr(model, "agg_layers") and len(model.agg_layers) > 0:
        last = model.agg_layers[-1]
        return getattr(last, "edge_layer", last)
    raise AttributeError(
        "Model does not expose an edge attention layer."
    )


def _construct_edge_scopes(layer, edge_batch):
    if hasattr(layer, "construct_b_scope"):
        return layer.construct_b_scope(edge_batch)
    if hasattr(layer, "_construct_edge_scopes"):
        return layer._construct_edge_scopes(edge_batch)
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


def extract_node_weights(model, batch_data, device):
    _run_attention_forward(model, batch_data)
    attn_module = _resolve_node_attention(model)
    if attn_module is None:
        raise AttributeError(
            "Model does not expose node attention weights."
        )
    attn = getattr(attn_module, "last_attn_weights", None)
    if attn is None:
        raise AttributeError(
            "Node attention weights were not stored during the forward pass."
        )

    return attn.detach()


def extract_bond_vecs(model, batch_data, device):
    layer = _resolve_edge_layer(model)
    bond_attn = getattr(layer, "bond_attn", None)
    if bond_attn is None:
        raise AttributeError(
            "Model does not expose edge attention weights."
        )
    edge_attr = batch_data.edge_attr.to(device)
    if getattr(layer, "edge_proj_adjust", None) is not None:
        edge_attr = layer.edge_proj_adjust(edge_attr)
    bond_message = layer.edge_proj(edge_attr)

    row, col = batch_data.edge_index
    row = row.to(device)
    col = col.to(device)
    e_batch = batch_data.batch.index_select(
        0, row).to(device)
    scopes, idx = _construct_edge_scopes(layer, e_batch)
    if idx is not None:
        bond_message = bond_message[idx]

    segments = []
    lengths = []
    for b, l in scopes:
        if l <= 0:
            continue
        segments.append(bond_message[b:b + l])
        lengths.append(int(l))
    if not segments:
        return torch.zeros(
            int(batch_data.x.size(0)),
            device=device,
            dtype=bond_message.dtype,
        )

    padded = torch.nn.utils.rnn.pad_sequence(
        segments, batch_first=True
        )
    B, L, _ = padded.size()
    lengths_t = torch.tensor(lengths, device=padded.device)
    mask = (torch.arange(
        L, device=padded.device).unsqueeze(0)
        < lengths_t.unsqueeze(1)
        )
    b_q = bond_attn.W_b_q(padded).view(
        B, L, bond_attn.heads, bond_attn.att_size
        ).transpose(1, 2)  # [B, H, L, D]
    mask_f = mask.unsqueeze(1).unsqueeze(-1).to(b_q.dtype)
    alpha = (b_q * bond_attn.weight_alpha.unsqueeze(0).unsqueeze(2)
        * bond_attn.scale_factor
        )
    alpha = F.softmax(alpha, dim=-1)
    global_q = (alpha * b_q * mask_f).sum(dim=2)  
    score = (b_q * global_q.unsqueeze(2)).sum(dim=-1) 
    score = score.masked_fill(
        ~mask.unsqueeze(1), float("-inf"))
    w = F.softmax(score, dim=-1).mean(dim=1) 
    w = w * mask.to(w.dtype)

    w_sorted = torch.cat(
        [w[b, :lengths[b]] for b in range(B)], dim=0
        )
    if idx is not None:
        inv_idx = torch.empty_like(idx)
        inv_idx[idx] = torch.arange(
            idx.numel(), device=idx.device
            )
        e_w = w_sorted[inv_idx]
    else:
        e_w = w_sorted

    num_nodes = int(batch_data.x.size(0))
    node_w = torch.zeros(
        num_nodes, device=device, dtype=e_w.dtype)
    half = e_w * 0.5
    node_w.scatter_add_(0, row, half)
    node_w.scatter_add_(0, col, half)

    return node_w.detach()
