import copy
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch.quasirandom import SobolEngine

from rules import feature_slices
from augmentations import apply_rulebook
from predictor import predict


def mask_groups(
    data_raw, idx,
    seed=None,
    all_candidates=False):

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    candidates = apply_rulebook(
        data_raw,
        idx,
        families=[
            "SP2_POLAR_ALL",
            "SP3_POLAR_ALL",
            "SP2_APOLAR_ALL",
            "SP3_APOLAR_ALL",
            "SP2_REACTIVE_ALL",
            "SP3_REACTIVE_ALL",
            "REDOX_FAMILY",
            "ACYL_FAMILY_ALL",
            "CARBAMATE_FAMILY_ALL",
            "AMIDE_FAMILY_ALL",
            "SULFURE_FAMILY_ALL",
            "PHOSPHORUS_FAMILY_ALL",
            "TOGGLE_CHARGE_FAMILY_ALL",
            "POLYVALENT_FAMILY_ALL",
            "TOGGLE_RING_FAMILY_ALL",
            "RING_FAMILY_ALL",
            "BOND_FAMILY_ALL",
            "DIARYL_FAMILY_ALL"
            ]
        )
    if not candidates:
        if all_candidates:
            return [(copy.deepcopy(data_raw), [idx])]
        return copy.deepcopy(data_raw), [idx]

    if not all_candidates:
        def rule_family(rule_id):
            parts = str(rule_id).split("_")
            if "ALL" in parts:
                idx = parts.index("ALL")
                return "_".join(parts[:idx + 1])
            if "FAMILY" in parts:
                idx = parts.index("FAMILY")
                return "_".join(parts[:idx + 1])
            return parts[0]
        buckets = {}
        for cand in candidates:
            fam = rule_family(cand[2])
            buckets.setdefault(fam, []).append(cand)
        fam_choice = random.choice(list(buckets.keys()))
        data, removed, _rule = random.choice(buckets[fam_choice])
        candidates = [(data, removed, _rule)]
    results = []
    for data, removed, _rule in candidates:
        n0 = data_raw.x.size(0)
        n1 = data.x.size(0)
        group = set()
        changed = []
        changed_adj = []
        if removed:
            group.update(removed)
            if not group:
                group.add(idx)
        else:
            slices = feature_slices()
            a_start, a_end = slices["atomic"]
            min_n = min(n0, n1)
            for j in range(min_n):
                if not torch.equal(
                    data_raw.x[j, a_start:a_end],
                    data.x[j, a_start:a_end]
                ):
                    changed.append(j)
            if n0 == n1:
                def to_adj(edge_index, n):
                    adj = [set() for _ in range(n)]
                    for i, j in edge_index.t().tolist():
                        adj[i].add(j)
                        adj[j].add(i)
                    return adj
                adj_raw = to_adj(data_raw.edge_index, n0)
                adj_new = to_adj(data.edge_index, n1)
                for j in range(min_n):
                    if adj_raw[j] != adj_new[j]:
                        changed.append(j)
                        changed_adj.append(j)
            group.update(changed)
            if not group:
                group.add(idx)
        results.append((data, sorted(group)))
    if all_candidates:
        return results
    return results[0]


def leave_one_group(
    model, raw, device,
    num_perturb=10,
    task_idx=0):

    base = copy.deepcopy(raw)
    if hasattr(raw, "y") and raw.y is not None:
        y = raw.y
        if y.dim() == 1:
            y = y.unsqueeze(0)
        base.y = y.to(device)
    else:
        tt = getattr(model, "task_type", None)
        if tt is not None:
            num_tasks = int(tt.numel())
        else:
            num_tasks = getattr(
                model, "num_tasks", None)
        if num_tasks is None:
            raise ValueError(
                "Cannot infer num_tasks for explanations."
                )
        base.y = torch.zeros(
            (1, num_tasks), 
            dtype=torch.float, 
            device=device
            )
    p0_all, _, _ = predict(
        model,
        DataLoader([base], batch_size=1),
        device,
        return_embeddings=False
        )
    p0 = p0_all[0, task_idx]
    n = raw.x.size(0)
    mean = np.zeros(n)
    std = np.zeros(n)
    pos = np.zeros(n)
    neg = np.zeros(n)
    trim = 0.1
    per_node = [[] for _ in range(n)]

    for i in range(n):
        sobol = SobolEngine(1, scramble=True)
        seq = sobol.draw(num_perturb).squeeze()
        for v in seq:
            seed = int(v.item() * (2**32 - 1))
            all_masks = mask_groups(
                raw, i, seed=seed, all_candidates=True)
            for masked, group_nodes in all_masks:
                if (torch.equal(raw.x, masked.x)
                        and torch.equal(
                            raw.edge_index, masked.edge_index)
                        and torch.equal(
                            raw.edge_attr, masked.edge_attr)):
                    continue
                if hasattr(raw, "y") and raw.y is not None:
                    y_local = raw.y
                    if y_local.dim() == 1:
                        y_local = y_local.unsqueeze(0)
                    masked.y = y_local.to(device)
                else:
                    masked.y = torch.zeros(
                        (1, num_tasks), dtype=torch.float,
                        device=device
                        )
                pred2_all, _, _ = predict(
                    model, DataLoader([masked], batch_size=1),
                    device, return_embeddings=False
                    )
                p2 = pred2_all[0, task_idx]
                delta = float(p0 - p2)
                if not group_nodes:
                    group_nodes = [i]
                share = delta / max(len(group_nodes), 1)
                for node_idx in group_nodes:
                    if node_idx < n:
                        per_node[node_idx].append(share)

    for i in range(n):
        arr = np.array(per_node[i], dtype=float)
        if arr.size == 0:
            continue
        k = int(len(arr) * trim)
        if len(arr) > 2 * k:
            arr = np.sort(arr)[k:-k]
        mean[i] = arr.mean()
        std[i] = arr.std(ddof=0)
        pos[i] = np.mean(arr > 0)
        neg[i] = np.mean(arr < 0)

    return mean, std, pos, neg