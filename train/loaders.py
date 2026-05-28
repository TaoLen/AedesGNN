import torch
from torch_geometric.data import Batch, Data
from torch.utils.data import DataLoader

from contrastive import (
    ContrastiveGraphDataset,
    collate_contrastive_graphs,
)
from graphs import graph2dataset
from utils import set_seed


contrastive_kwargs = {
    "node_mask_ratio": 0.25,
    "edge_mask_ratio": 0.25,
    "min_fragment_size": 4,
}


def _normalize_smiles_set(smiles_iterable):
    normalized = set()
    if smiles_iterable is None:
        return normalized
    for smi in smiles_iterable:
        if smi is None:
            continue
        smi = str(smi).strip()
        if not smi or smi.lower() == "nan":
            continue
        normalized.add(smi)
    return normalized


def _assert_no_split_overlap(train_smiles, val_smiles, test_smiles):
    train_set = _normalize_smiles_set(train_smiles)
    val_set = _normalize_smiles_set(val_smiles)
    test_set = _normalize_smiles_set(test_smiles)

    overlaps = {
        "train/val": train_set & val_set,
        "train/test": train_set & test_set,
        "val/test": val_set & test_set,
    }
    leaking = {
        pair: values for pair, values in overlaps.items() if values
    }
    if leaking:
        details = ", ".join(
            f"{pair}={len(values)}" for pair, values in leaking.items()
        )
        raise ValueError(
            "SMILES overlap detected across data splits. "
            f"This would cause train/validation/test leakage: {details}"
        )


def _collate_labels(labels):
    tensors = []
    for label in labels:
        if label is None:
            continue
        tensor = label
        if not torch.is_tensor(tensor):
            tensor = torch.tensor(tensor, dtype=torch.float32)
        if tensor.dim() > 1 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        tensors.append(tensor)

    if not tensors:
        return None

    stacked = torch.stack(tensors, dim=0)
    if stacked.dim() == 1:
        stacked = stacked.unsqueeze(1)
    return stacked


def _unwrap_batch(batch):
    if isinstance(batch, dict):
        return batch.get("supervised", batch.get("original"))
    return batch


def collate_graphs(samples):
    samples = [s for s in samples 
        if s is not None and hasattr(s, 'y')]
    if len(samples) == 0:
        return None

    graphs = [s for s in samples if isinstance(s, Data)]
    batch = Batch.from_data_list(graphs)
    batch.smiles = [s.smiles for s in samples]
    labels = [s.y for s in samples]
    batch.y = _collate_labels(labels)

    return batch


def graph_loader(
    train_smiles,
    val_smiles,
    test_smiles,
    y_train=None,
    y_val=None,
    y_test=None,
    batch_size=32,
    seed=None,
    contrastive=False,
    contrastive_eval=False,
    contrastive_kwargs=None,
    check_split_overlap=True):

    if seed is not None:
        set_seed(seed)
    if check_split_overlap:
        _assert_no_split_overlap(
            train_smiles,
            val_smiles,
            test_smiles,
        )

    train_dataset = graph2dataset(train_smiles, y_train)
    val_dataset   = graph2dataset(val_smiles,   y_val)
    test_dataset  = graph2dataset(test_smiles,  y_test)

    # This module relies on custom collate functions to preserve the
    # repository's graph construction contract and, in contrastive mode,
    # to materialize fragment_atom_index / fragment_index / num_fragments.
    train_collate_fn = collate_graphs
    val_collate_fn = collate_graphs
    if contrastive:
        train_dataset = ContrastiveGraphDataset(
            train_dataset,
            **(contrastive_kwargs or {})
            )
        train_collate_fn = collate_contrastive_graphs
    if contrastive_eval:
        val_dataset = ContrastiveGraphDataset(
            val_dataset,
            **(contrastive_kwargs or {})
            )
        val_collate_fn = collate_contrastive_graphs

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,    
        collate_fn=train_collate_fn
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_collate_fn
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graphs
        )

    return train_loader, val_loader, test_loader


def graph_info(data_loader):
    node_feature_dim = None
    edge_feature_dim = None
    num_tasks = None

    for batch in data_loader:
        batch = _unwrap_batch(batch)
        if batch is None:
            continue
        if isinstance(batch, Batch):
            if batch.y is not None and batch.y.size(0) > 0:
                num_tasks = batch.y.size(1
                    ) if batch.y.ndim > 1 else 1
            node_feature_dim = batch.x.size(1)
            edge_feature_dim = batch.edge_attr.size(1)
        else:
            if batch.y is not None and batch.y.size(0) > 0:
                num_tasks = batch.y.size(1
                    ) if batch.y.ndim > 1 else 1
            node_feature_dim = batch.x.size(1)
            edge_feature_dim = batch.edge_attr.size(1)

        break

    return node_feature_dim, edge_feature_dim, num_tasks


def infer_task_metadata(data_loader, tol=1e-4):
    num_tasks = None
    cond_in01 = None
    cond_near01 = None
    has_valid = None
    integer_ok = None
    class_sets = None

    for batch in data_loader:
        batch = _unwrap_batch(batch)
        if batch is None:
            continue
        if not hasattr(batch, "y") or batch.y is None:
            continue

        y = batch.y
        if y.dim() == 1:
            y = y.unsqueeze(1)
        if num_tasks is None:
            num_tasks = y.size(1)
            cond_in01 = torch.ones(num_tasks, dtype=torch.bool)
            cond_near01 = torch.ones(num_tasks, dtype=torch.bool)
            has_valid = torch.zeros(num_tasks, dtype=torch.bool)
            integer_ok = torch.ones(num_tasks, dtype=torch.bool)
            class_sets = [set() for _ in range(num_tasks)]

        mask = ~torch.isnan(y)
        yv = torch.nan_to_num(y, nan=0.0)

        for j in range(num_tasks):
            valid = mask[:, j]
            if not valid.any():
                continue
            has_valid[j] = True
            vals = yv[valid, j]
            if cond_in01[j]:
                diff_in01 = (
                    vals - vals.clamp(0.0, 1.0)
                ).abs().max().item()
                if diff_in01 > tol:
                    cond_in01[j] = False
            if cond_near01[j]:
                y_round01 = (vals >= 0.5).float()
                diff_near01 = (
                    vals - y_round01
                ).abs().max().item()
                if diff_near01 > tol:
                    cond_near01[j] = False

            if integer_ok[j]:
                rounded = vals.round()
                if (vals - rounded).abs().max().item() > tol:
                    integer_ok[j] = False
                else:
                    class_sets[j].update(
                        rounded.to(torch.long).cpu().tolist()
                        )

    if num_tasks is None:
        return None, None, None

    task_type = torch.zeros(
        num_tasks, dtype=torch.long
        )
    is_binary = cond_in01 & cond_near01 & has_valid
    task_type[is_binary] = 1

    mc_class_counts = torch.zeros(
        num_tasks, dtype=torch.long
        )
    mc_label_values = [None] * num_tasks
    for j in range(num_tasks):
        if task_type[j] != 0:
            continue
        if not has_valid[j]:
            continue
        if not integer_ok[j]:
            continue
        if len(class_sets[j]) < 3:
            continue

        task_type[j] = 2
        values = sorted(class_sets[j])
        mc_label_values[j] = values
        mc_class_counts[j] = len(values)

    return task_type, mc_class_counts, mc_label_values
