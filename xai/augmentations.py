import copy
import torch
import networkx as nx

from graphs import ATOMIC_NUMBER
from utils import one_hot
from rules import (
    CHARGE_CATEGORIES,
    HYBRIDIZATION_CATEGORIES,
    FRAGMENTS,
    MAX_VALENCE,
    bond_orders,
    node_valence_degree,
    atomic_num_at,
    atomic_charge_at,
    adjacency_list,
    bond_order_between,
    branch_nodes,
    match_fragment,
    sp2_polar_rules,
    sp3_polar_rules,
    sp2_apolar_rules,
    sp3_apolar_rules,
    sp3_reactive_rules,
    sp2_reactive_rules,
    acyl_family_rules,
    amide_family_rules,
    carbamate_family_rules,
    sulfure_family_rules,
    redox_family_rules,
    toggle_ring_family_rules,
    ring_family_rules,
    bond_family_rules,
    diaryl_family_rules,
    toggle_charge_family_rules,
    aliphatic_family_rules,
    polyvalent_family_rules,
    rule_fragments,
    sp2_polar_filters,
    is_sp2_attachment,
    is_sp3_attachment,
    adjacent_to_carbonyl,
    is_acyl_hetero_link,
    is_cf3_f,
    is_amidine_core,
    valence_ok,
    violates_blacklist,
    match_acyl_from_fragment,
    feature_slices,
    is_ring_atom
    )


def set_degree(
    data, 
    node_idx, 
    degree_val):

    atoms = ATOMIC_NUMBER()
    d_start = len(atoms)
    d_end = d_start + 8
    degree_val = int(min(
        max(degree_val, 0), 7))
    data.x[node_idx, 
        d_start:d_end] = torch.tensor(
        one_hot(degree_val, list(range(8))),
        dtype=data.x.dtype,
        device=data.x.device
        )


def set_hybridization(
    data, 
    node_idx, 
    hybrid_type):
    
    atoms = ATOMIC_NUMBER()
    d_start = len(atoms)
    c_start = d_start + 8
    h_start = c_start + len(CHARGE_CATEGORIES)
    h_end = h_start + len(HYBRIDIZATION_CATEGORIES)
    idx = HYBRIDIZATION_CATEGORIES.index(hybrid_type)
    data.x[node_idx, h_start:h_end] = torch.tensor(
        one_hot(idx, list(range(len(
            HYBRIDIZATION_CATEGORIES)))),
        dtype=data.x.dtype,
        device=data.x.device
        )


def set_aromatic(
    data, 
    node_idx, 
    is_aromatic):
    
    atoms = ATOMIC_NUMBER()
    d_start = len(atoms)
    c_start = d_start + 8
    h_start = c_start + len(CHARGE_CATEGORIES)
    arom_idx = h_start + len(HYBRIDIZATION_CATEGORIES)
    data.x[node_idx, arom_idx] = float(is_aromatic)


def update_nodes_from_edges(data, node_idx):
    if data.edge_attr is None or data.edge_attr.numel() == 0:
        set_degree(data, node_idx, 0)
        set_hybridization(
            data, node_idx, HYBRIDIZATION_CATEGORIES[2])
        set_aromatic(data, node_idx, False)
        return
    src, dst = data.edge_index
    orders = bond_orders(data.edge_attr)
    mask = (src == node_idx) | (dst == node_idx)
    degree = int(mask.sum().item())
    set_degree(data, node_idx, degree)
    if not mask.any():
        set_hybridization(
            data, node_idx, HYBRIDIZATION_CATEGORIES[2])
        set_aromatic(data, node_idx, False)
        return
    ords = orders[mask].tolist()
    aromatic = any(abs(o - 1.5) < 1e-6 for o in ords)
    if any(abs(o - 3.0) < 1e-6 for o in ords):
        hybrid = HYBRIDIZATION_CATEGORIES[0]
    elif any(abs(o - 2.0) < 1e-6 for o in ords) or aromatic:
        hybrid = HYBRIDIZATION_CATEGORIES[1]
    else:
        hybrid = HYBRIDIZATION_CATEGORIES[2]
    set_hybridization(data, node_idx, hybrid)
    set_aromatic(data, node_idx, aromatic)


def update_all_features(data):
    for i in range(data.x.size(0)):
        _, deg = node_valence_degree(data, i)
        set_degree(data, i, deg)
        update_nodes_from_edges(data, i)


def set_formal_charge(data, node_idx, charge_val):
    atoms = ATOMIC_NUMBER()
    d_start = len(atoms)
    c_start = d_start + 8
    c_end = c_start + len(CHARGE_CATEGORIES)
    data.x[node_idx, c_start:c_end] = torch.tensor(
        one_hot(charge_val, 
        CHARGE_CATEGORIES),
        dtype=data.x.dtype,
        device=data.x.device
        )


def edge_attr_template(edge_attr, order=1.0, 
        ring=False, conjugated=False):
    size = edge_attr.size(1)
    attr = torch.zeros(size, 
            dtype=edge_attr.dtype, 
            device=edge_attr.device
            )
    if order == 1.0:
        attr[0] = 1.0
    elif order == 2.0:
        attr[1] = 1.0
    elif order == 3.0:
        attr[2] = 1.0
    else:
        attr[3] = 1.0
    if size > 4:
        if order == 1.5:
            attr[4] = 1.0
        else:
            attr[4] = 1.0 if conjugated else 0.0
    if size > 5:
        attr[5] = 1.0 if ring else 0.0
    if size > 6:
        attr[6] = 1.0

    return attr


def normalize_edge_attr(data):
    if data.edge_attr is None or data.edge_attr.numel() == 0:
        return
    size = data.edge_attr.size(1)
    bond_type = data.edge_attr[:, :4]
    is_arom = bond_type[:, 3] > 0.5

    if size > 4:
        conj = is_arom.clone()
        if size > 1:
            orders = bond_orders(data.edge_attr)
            src, dst = data.edge_index
            for e_idx, (i, j) in enumerate(
                zip(src.tolist(), dst.tolist())):
                if is_arom[e_idx]:
                    continue
                if orders[e_idx] < 2.0:
                    continue
                mask_i = (src == i) | (dst == i)
                mask_j = (src == j) | (dst == j)
                if any(float(o.item()
                    ) >= 2.0 for o in orders[mask_i]) and any(
                    float(o.item()) >= 2.0 for o in orders[mask_j]):
                    conj[e_idx] = True
        data.edge_attr[:, 4] = conj.float()

    if size > 6:
        data.edge_attr[:, 6:] = 0.0
        data.edge_attr[:, 6] = 1.0


def remove_nodes(data, remove_set):
    keep = [i for i in range(
        data.x.size(0)) if i not in remove_set]
    mapping = {old: new for new, old in enumerate(keep)}
    data.x = data.x[keep].clone()
    src, dst = data.edge_index
    keep_edges = []
    new_edges = []

    for e_idx, (i, j) in enumerate(
        zip(src.tolist(), dst.tolist())):

        if i in mapping and j in mapping:
            keep_edges.append(e_idx)
            new_edges.append([mapping[i], mapping[j]])
    if keep_edges:
        data.edge_index = torch.tensor(
            new_edges, dtype=torch.long, 
            device=data.edge_index.device
            ).t().contiguous()
        data.edge_attr = data.edge_attr[keep_edges].clone()
    else:
        data.edge_index = torch.zeros(
            (2, 0), dtype=torch.long)
        data.edge_attr = torch.zeros(
            (0, data.edge_attr.size(1)), 
            dtype=data.edge_attr.dtype
            )
    return mapping


def add_nodes(data, nodes):
    if not nodes:
        return []
    device = data.x.device
    new = torch.stack(nodes).to(device)
    start = data.x.size(0)
    data.x = torch.cat([data.x, new], dim=0)
    return list(range(start, start + new.size(0)))


def add_edge(data, i, j, order=1.0, 
        ring=False, conjugated=False):
    attr = edge_attr_template(
        data.edge_attr, order=order, 
        ring=ring, conjugated=conjugated
        )
    edge_index = torch.tensor(
        [[i, j]], dtype=torch.long, 
        device=data.edge_index.device
        ).t()
    data.edge_index = torch.cat(
        [data.edge_index, edge_index], dim=1)
    data.edge_attr = torch.cat(
        [data.edge_attr, attr.view(1, -1)], dim=0)
    normalize_edge_attr(data)


def set_atomic_num(data, node_idx, atom_num):
    atoms = ATOMIC_NUMBER()
    a_start = 0
    a_end = len(atoms)
    data.x[node_idx, a_start:a_end] = torch.tensor(
        one_hot(atom_num, atoms),
        dtype=data.x.dtype,
        device=data.x.device
        )


def make_atom_feature(atom_num):
    atoms = ATOMIC_NUMBER()
    degree = 0
    charge = 0
    hybrid = HYBRIDIZATION_CATEGORIES[2]
    feat = (
        one_hot(atom_num, atoms)
        + one_hot(degree, list(range(8)))
        + one_hot(charge, CHARGE_CATEGORIES)
        + one_hot(hybrid, HYBRIDIZATION_CATEGORIES)
        + [0.0])
    
    return torch.tensor(
        feat, dtype=torch.float
        )


def replace_fragment(
    data, attachment, neighbor, 
    frag_key, copy_data=True):

    frag = FRAGMENTS.get(frag_key)
    if frag is None:
        return None, []
    removed = branch_nodes(
        data, attachment, neighbor)
    attach_atom = atomic_num_at(
        data, attachment)
    valence, degree = node_valence_degree(
        data, attachment)
    old_order = bond_order_between(
        data, attachment, neighbor)
    if old_order is None:
        return None, []
    new_valence = valence - old_order + 1.0
    if new_valence - 1e-6 > MAX_VALENCE.get(attach_atom, 4):
        return None, []
    if attach_atom == 8:
        if degree > 2:
            return None, []
        src, dst = data.edge_index
        orders = bond_orders(data.edge_attr)
        mask = (src == attachment) | (dst == attachment)
        if any(float(o.item()) >= 2.0 for o in orders[mask]):
            return None, []
    if attach_atom in (16, 34):
        if degree > 2:
            return None, []
    if copy_data:
        data = copy.deepcopy(data)
    mapping = remove_nodes(data, removed)
    attach_new = mapping[attachment]
    nodes = [make_atom_feature(atom) for atom in frag["atoms"]]
    new_ids = add_nodes(data, nodes)
    attach_idx = new_ids[frag["attach_idx"]]
    add_edge(data, attach_new, attach_idx, order=1.0)
    for i, j, order in frag["edges"]:
        add_edge(data, new_ids[i], new_ids[j], order=order)
    update_all_features(data)
    normalize_edge_attr(data)

    changed_nodes = set()
    shift = sum(1 for r in removed if r < attachment)
    attach_new = attachment - shift
    changed_nodes.add(attach_new)
    frag_len = len(frag.get("atoms", []))
    if frag_len > 0:
        n_new = data.x.size(0)
        changed_nodes.update(range(n_new - frag_len, n_new))
    for i in changed_nodes:
        atom_num = atomic_num_at(data, i)
        valence, _ = node_valence_degree(data, i)
        if frag_key in ("NO2", "N3"
            ) and atom_num == 7 and valence <= 4.0 + 1e-6:
            continue
        if not valence_ok(atom_num, valence):
            return None, []
    if frag_key == "NO2":
        frag_len = len(frag.get("atoms", []))
        n_new = data.x.size(0)
        new_nodes = list(range(max(n_new - frag_len, 0), n_new))
        n_nodes = [n for n in new_nodes if atomic_num_at(data, n) == 7]
        if n_nodes:
            n_idx = n_nodes[0]
            adj_local = adjacency_list(data.edge_index, data.x.size(0))
            o_nodes = [
                n for n in adj_local[n_idx]
                if atomic_num_at(data, n) == 8
                ]
            if len(o_nodes) >= 2:
                set_formal_charge(data, n_idx, 1)
                set_formal_charge(data, o_nodes[0], -1)
                set_formal_charge(data, o_nodes[1], -1)
            else:
                pass
    if frag_key in ("N3", "N2"):
        frag_len = len(frag.get("atoms", []))
        n_new = data.x.size(0)
        new_nodes = list(range(max(n_new - frag_len, 0), n_new))
        n_nodes = [n for n in new_nodes if atomic_num_at(data, n) == 7]
        if frag_key == "N2":
            if len(n_nodes) >= 2:
                set_formal_charge(data, n_nodes[0], 1)
                set_formal_charge(data, n_nodes[1], -1)
        else:
            if len(n_nodes) >= 3:
                set_formal_charge(data, n_nodes[1], 1)
                set_formal_charge(data, n_nodes[2], -1)
    if changed_nodes:
        data_chk = copy.copy(data)
        data_chk.x = data.x.clone()
        data_chk.edge_index = data.edge_index.clone()
        data_chk.edge_attr = data.edge_attr.clone()
        remove_nodes(
            data_chk,
            set(range(data_chk.x.size(0))) - set(changed_nodes)
            )
        if violates_blacklist(data_chk):
            return None, []
  
    return data, list(removed)


def apply_polar_families(
    families,
    attachment_sp2,
    attachment_sp3,
    apply_family):

    if "SP2_POLAR_ALL" in families:
        apply_family(
            sp2_polar_rules(),
            attachment_sp2,
            require_aromatic=True,
            block_if_reactive=True
            )
    if "SP3_POLAR_ALL" in families:
        apply_family(
            sp3_polar_rules(),
            attachment_sp3,
            block_if_sp2_polar=True,
            block_if_reactive=True
            )


def apply_apolar_families(
    families,
    attachment_sp2,
    attachment_sp3,
    attachment_aromatic,
    node_idx,
    adj,
    data_raw,
    apply_family):

    sp2_apolar_touched = set()
    if "SP2_APOLAR_ALL" in families:
        apply_family(
            sp2_apolar_rules(),
            attachment_sp2,
            require_aromatic=True,
            block_if_reactive=True
            )
        if attachment_sp2 and attachment_aromatic:
            for nbr in adj[node_idx]:
                for rule in sp2_apolar_rules():
                    left, right = rule_fragments(rule["direction"])
                    if not left or not right:
                        continue
                    if not match_fragment(data_raw, node_idx, nbr, left):
                        continue
                    data_work = copy.copy(data_raw)
                    data_work.x = data_raw.x.clone()
                    data_work.edge_index = data_raw.edge_index.clone()
                    data_work.edge_attr = data_raw.edge_attr.clone()
                    new_data, _ = replace_fragment(
                        data_work, node_idx, nbr, right, copy_data=False
                        )
                    if new_data is not None:
                        sp2_apolar_touched.add(nbr)

    if "SP3_APOLAR_ALL" in families:
        apply_family(
            sp3_apolar_rules(),
            attachment_sp3,
            exclude_nodes=sp2_apolar_touched,
            block_if_sp2_apolar=True,
            block_if_reactive=True
            )
    return sp2_apolar_touched


def apply_reactive_families(
    families,
    attachment_sp2,
    attachment_sp3,
    apply_family):

    if "SP2_REACTIVE_ALL" in families:
        apply_family(
            sp2_reactive_rules(),
            attachment_sp2,
            block_adjacent_carbonyl=False
            )
    if "SP3_REACTIVE_ALL" in families:
        apply_family(
            sp3_reactive_rules(),
            attachment_sp3,
            block_adjacent_carbonyl=False
            )


def apply_redox_family(
    families,
    data_raw,
    node_idx,
    adj,
    out,
    seen):

    if "REDOX_FAMILY" not in families:
        return

    def carbonyl_neighbors(data, carbonyl):
        adj_local = adjacency_list(
            data.edge_index, data.x.size(0))
        hetero_double = None
        single_c = []
        single_hetero = []
        for nbr in adj_local[carbonyl]:
            order = bond_order_between(
                data, carbonyl, nbr)
            atom = atomic_num_at(data, nbr)
            if order in (2.0, 1.5
                ) and atom in (7, 8, 16, 34):
                hetero_double = nbr
            elif order in (1.0, 1.5) and atom == 6:
                single_c.append(nbr)
            elif order in (1.0, 1.5
                ) and atom in (7, 8, 16, 34):
                single_hetero.append(nbr)
        return hetero_double, single_c, single_hetero

    def is_aromatic_carbon(data, node):
        slices = feature_slices()
        arom_idx = slices["aromatic"]
        return bool(data.x[node, arom_idx].item() > 0.5)

    def is_oh_like(data, oxygen, carbonyl):
        if atomic_num_at(data, oxygen) != 8:
            return False
        if bond_order_between(
            data, oxygen, carbonyl) != 1.0:
            return False
        adj_local = adjacency_list(
            data.edge_index, data.x.size(0))
        others = [n for n in adj_local[oxygen] if n != carbonyl]
        return len(others) == 0

    def is_sh_like(data, sulfur, carbonyl):
        if atomic_num_at(data, sulfur) != 16:
            return False
        if bond_order_between(
            data, sulfur, carbonyl) != 1.0:
            return False
        adj_local = adjacency_list(
            data.edge_index, data.x.size(0))
        others = [n for n in adj_local[sulfur] if n != carbonyl]
        return len(others) == 0

    def is_alcohol(data, carbonyl, oxygen):
        if not is_oh_like(
            data, oxygen, carbonyl):
            return False
        _, single_c, single_hetero = carbonyl_neighbors(data, carbonyl)
        single_hetero = [n for n in single_hetero if n != oxygen]
        if single_hetero:
            return False
        return len(single_c) >= 1

    def is_thiol(data, carbonyl, sulfur):
        if not is_sh_like(data, sulfur, carbonyl):
            return False
        _, single_c, single_hetero = carbonyl_neighbors(data, carbonyl)
        single_hetero = [n for n in single_hetero if n != sulfur]
        if single_hetero:
            return False
        return len(single_c) >= 1

    def is_selenol(data, carbonyl, selenium):
        if atomic_num_at(data, selenium) != 34:
            return False
        if bond_order_between(
            data, selenium, carbonyl) != 1.0:
            return False
        adj_local = adjacency_list(
            data.edge_index, data.x.size(0))
        others = [n for n in adj_local[selenium] if n != carbonyl]
        if len(others) != 0:
            return False
        _, single_c, single_hetero = carbonyl_neighbors(data, carbonyl)
        single_hetero = [n for n in single_hetero if n != selenium]
        if single_hetero:
            return False
        return len(single_c) >= 1

    def has_hetero_neighbors(data, carbonyl, ignore):
        adj_local = adjacency_list(
            data.edge_index, data.x.size(0))
        for nbr in adj_local[carbonyl]:
            if nbr == ignore:
                continue
            atom = atomic_num_at(data, nbr)
            if atom in (7, 8, 16, 34):
                return True
        return False

    def carbonyl_or_hetero_idx(data, node_idx_local):
        for nbr in adj[node_idx_local]:
            order = bond_order_between(
                data, node_idx_local, nbr)
            atom = atomic_num_at(data, nbr)
            if order in (2.0, 1.5) and atom in (7, 8, 16, 34):
                return node_idx_local, nbr
            if order == 1.0 and atom in (8, 16, 34):
                if is_alcohol(data, nbr, node_idx_local):
                    return nbr, node_idx_local
                if is_thiol(data, nbr, node_idx_local):
                    return nbr, node_idx_local
                if is_selenol(data, nbr, node_idx_local):
                    return nbr, node_idx_local
        return None, None

    redox_rules = redox_family_rules()
    cores = []
    c0, h0 = carbonyl_or_hetero_idx(
        data_raw, node_idx)
    if c0 is not None:
        cores.append((c0, h0))
    for nbr in adj[node_idx]:
        c0, h0 = carbonyl_or_hetero_idx(
            data_raw, nbr)
        if c0 is not None:
            cores.append((c0, h0))
    cores = list(dict.fromkeys(cores))
    out_before = len(out)
    for carbonyl, hetero in cores:
        if carbonyl is None or hetero is None:
            continue
        if is_aromatic_carbon(
            data_raw, carbonyl):
            continue
        if has_hetero_neighbors(
            data_raw, carbonyl, hetero):
            continue
        for rule in redox_rules:
            left, right = rule_fragments(rule["direction"])
            if not left or not right:
                continue
            if left in ("alcohol", "thiol", "selenol"):
                if left == "alcohol" and not is_alcohol(
                    data_raw, carbonyl, hetero):
                    continue
                if left == "thiol" and not is_thiol(
                    data_raw, carbonyl, hetero):
                    continue
                if left == "selenol" and not is_selenol(
                    data_raw, carbonyl, hetero):
                    continue
            else:
                if not match_fragment(
                    data_raw, carbonyl, hetero, left):
                    continue
            data_work = copy.copy(data_raw)
            data_work.x = data_raw.x.clone()
            data_work.edge_index = data_raw.edge_index.clone()
            data_work.edge_attr = data_raw.edge_attr.clone()

            if left in ("alcohol", "thiol", "selenol"):
                if right == "ketone":
                    set_atomic_num(data_work, hetero, 8)
                    set_bond_order(
                        data_work, carbonyl, hetero, 2.0)
                elif right == "thioketone":
                    set_atomic_num(data_work, hetero, 16)
                    set_bond_order(
                        data_work, carbonyl, hetero, 2.0)
                elif right == "selenoketone":
                    set_atomic_num(data_work, hetero, 34)
                    set_bond_order(
                        data_work, carbonyl, hetero, 2.0)
            else:
                if right == "alcohol":
                    set_atomic_num(data_work, hetero, 8)
                    set_bond_order(
                        data_work, carbonyl, hetero, 1.0)
                elif right == "thiol":
                    set_atomic_num(data_work, hetero, 16)
                    set_bond_order(
                        data_work, carbonyl, hetero, 1.0)
                elif right == "selenol":
                    set_atomic_num(data_work, hetero, 34)
                    set_bond_order(data_work, carbonyl, hetero, 1.0)
            update_all_features(data_work)
            normalize_edge_attr(data_work)
            for i in (carbonyl, hetero):
                atom_num = atomic_num_at(data_work, i)
                valence, _ = node_valence_degree(data_work, i)
                if not valence_ok(atom_num, valence):
                    break
            else:
                data_chk = copy.copy(data_work)
                data_chk.x = data_work.x.clone()
                data_chk.edge_index = data_work.edge_index.clone()
                data_chk.edge_attr = data_work.edge_attr.clone()
                remove_nodes(data_chk,
                    set(range(data_chk.x.size(0))) - set([carbonyl, hetero])
                    )
                if not violates_blacklist(data_chk):
                    h = (data_work.x.cpu().numpy().tobytes(),
                        data_work.edge_index.cpu().numpy().tobytes(),
                        data_work.edge_attr.cpu().numpy().tobytes()
                        )
                    if h not in seen:
                        seen.add(h)
                        out.append((data_work, 
                            [carbonyl, hetero], rule["rule_id"])
                            )
    if len(out) > out_before:
        pass


def apply_acyl_family(
    families,
    data_raw,
    node_idx,
    adj,
    attachment_sp2,
    attachment_sp3,
    out,
    seen):
    
    if "ACYL_FAMILY_ALL" not in families:
        return
    if not (attachment_sp2 or attachment_sp3):
        return

    def acyl_variants(frag_key):
        inv_map = {
            "COOMe": "COOMe_inv",
            "CONH2": "CONH2_inv",
            "thioester": "thioester_inv",
            "ketone": "ketone_inv",
            "thioketone": "thioketone_inv",
            "selenoketone": "selenoketone_inv"
            }
        variants = [(frag_key, False)]
        inv_key = inv_map.get(frag_key)
        if inv_key:
            variants.append((inv_key, True))
        return variants

    def acyl_replacement_key(right_key, inverted):
        if right_key in ("ketone", "thioketone", "selenoketone"):
            return f"{right_key}_inv" if inverted else f"{right_key}_sub"
        if inverted:
            if right_key == "COOMe":
                return "COOMe_inv"
            if right_key == "CONH2":
                return "CONH2_inv"
            if right_key == "thioester":
                return "thioester_inv"
            return None
        return right_key

    def set_bond_order(data, i, j, order):
        src, dst = data.edge_index
        mask = ((src == i) & (dst == j)) | ((src == j) & (dst == i))
        idxs = torch.where(mask)[0]
        if idxs.numel() == 0:
            return False
        e_idx = int(idxs[0].item())
        data.edge_attr[e_idx, :4] = 0.0
        if order == 1.0:
            data.edge_attr[e_idx, 0] = 1.0
        elif order == 2.0:
            data.edge_attr[e_idx, 1] = 1.0
        elif order == 3.0:
            data.edge_attr[e_idx, 2] = 1.0
        else:
            data.edge_attr[e_idx, 3] = 1.0
        normalize_edge_attr(data)
        return True

    def acyl_targets(right_key):
        if right_key in ("ketone",
                "ketone_sub", "ketone_inv"):
            return 8, 6
        if right_key in ("thioketone",
                "thioketone_sub", "thioketone_inv"):
            return 16, 6
        if right_key in ("selenoketone",
                "selenoketone_sub", "selenoketone_inv"):
            return 34, 6
        if right_key == "thioester":
            return 8, 16
        if right_key == "COOMe":
            return 8, 8
        if right_key == "CONH2":
            return 8, 7
        if right_key == "COOH":
            return 8, 8
        return None, None

    def acyl_core_substitute(data_raw_local,
            attach, nbr, inverted, right_key):

        if right_key == "COOH" and inverted:
            return None, None
        data_work = copy.copy(data_raw_local)
        data_work.x = data_raw_local.x.clone()
        data_work.edge_index = data_raw_local.edge_index.clone()
        data_work.edge_attr = data_raw_local.edge_attr.clone()
        adj_local = adjacency_list(
            data_work.edge_index, data_work.x.size(0))

        if inverted:
            x = nbr
            c = None
            for cand in adj_local[x]:
                if cand == attach:
                    continue
                if atomic_num_at(data_work, cand
                        ) == 6 and bond_order_between(
                    data_work, x, cand) == 1.0:
                    c = cand
                    break
        else:
            c = nbr
            x = None
            for cand in adj_local[c]:
                if cand == attach:
                    continue
                if bond_order_between(
                    data_work, c, cand) == 1.0:
                    x = cand
                    break
        if c is None or x is None:
            return None, None

        dbl = None
        for cand in adj_local[c]:
            if bond_order_between(
                data_work, c, cand) == 2.0 and atomic_num_at(
                data_work, cand) in (8, 16, 34):
                dbl = cand
                break
        if dbl is None:
            return None, None

        target_dbl, target_x = acyl_targets(right_key)
        if target_dbl is None or target_x is None:
            return None, None

        if right_key == "COOH":
            x_others = [n for n in adj_local[x] if n != c]
            if x_others:
                return None, None

        if atomic_num_at(data_work, dbl) != target_dbl:
            set_atomic_num(data_work, dbl, target_dbl)
        if atomic_num_at(data_work, x) != target_x:
            set_atomic_num(data_work, x, target_x)

        if not set_bond_order(data_work, c, dbl, 2.0):
            return None, None
        if not set_bond_order(data_work, c, x, 1.0):
            return None, None

        update_all_features(data_work)
        normalize_edge_attr(data_work)
        if target_x == 16:
            _, degree_s = node_valence_degree(data_work, x)
            if degree_s > 2:
                return None, None
            adj_check = adjacency_list(
                data_work.edge_index,
                data_work.x.size(0))
            non_carbonyl = [n for n in adj_check[x] if n != c]
            if len(non_carbonyl) > 1:
                return None, None
        for idx in (c, x, dbl):
            atom_num = atomic_num_at(data_work, idx)
            valence, _ = node_valence_degree(data_work, idx)
            if not valence_ok(atom_num, valence):
                return None, None
        data_chk = copy.copy(data_work)
        data_chk.x = data_work.x.clone()
        data_chk.edge_index = data_work.edge_index.clone()
        data_chk.edge_attr = data_work.edge_attr.clone()
        remove_nodes(
            data_chk, set(range(data_chk.x.size(0))
                ) - set([c, x, dbl]))
        if violates_blacklist(data_chk):
            return None, None

        return data_work, [c, x, dbl]

    out_before = len(out)
    for rule in acyl_family_rules():
        left, right = rule_fragments(rule["direction"])
        if not left or not right:
            continue
        for nbr in adj[node_idx]:
            match_key = None
            inverted = False
            for cand_key, cand_inverted in acyl_variants(left):
                if match_acyl_from_fragment(
                    data_raw, node_idx, nbr, cand_key):
                    match_key = cand_key
                    inverted = cand_inverted
                    break
            if match_key is None:
                continue
            replace_key = acyl_replacement_key(right, inverted)
            if replace_key is None:
                continue
            new_data, removed = acyl_core_substitute(
                data_raw, node_idx, 
                nbr, inverted, right
                )
            if new_data is None:
                continue
            h = (new_data.x.cpu().numpy().tobytes(),
                new_data.edge_index.cpu().numpy().tobytes(),
                new_data.edge_attr.cpu().numpy().tobytes()
                )
            if h in seen:
                continue
            seen.add(h)
            out.append((new_data, removed, rule["rule_id"]))
    if len(out) > out_before:
        pass


def apply_amide_family(
    families,
    data_raw,
    node_idx,
    attachment_sp2,
    attachment_sp3,
    out,
    seen):

    if "AMIDE_FAMILY_ALL" not in families:
        return
    if not (attachment_sp2 or attachment_sp3):
        return

    left_map = {
        "CONH2": (8, [6, 7]),
        "thioamide": (16, [6, 7]),
        "selenoamide": (34, [6, 7]),
        "urea": (8, [7, 7]),
        "thiourea": (16, [7, 7]),
        "selenourea": (34, [7, 7]),
        "guanidine": (7, [7, 7]),
        "amidine": (7, [6, 7])
        }
    out_before = len(out)

    if atomic_num_at(data_raw, node_idx) == 6:
        src, dst = data_raw.edge_index
        orders = bond_orders(data_raw.edge_attr)
        mask = (src == node_idx) | (dst == node_idx)
        hetero_double = None
        lateral = []
        other_neighbors = []
        for k in torch.where(mask)[0].tolist():
            nbr = int(dst[k].item()) if int(src[k].item()
                ) == node_idx else int(src[k].item())
            order = float(orders[k].item())
            atom = atomic_num_at(data_raw, nbr)
            if order == 2.0 and atom in (7, 8, 16, 34):
                hetero_double = nbr
            elif order == 1.0:
                lateral.append(nbr)
            else:
                other_neighbors.append(nbr)

        if hetero_double is not None and len(lateral) == 2:
            if not any(atomic_num_at(
                data_raw, n) != 6 for n in other_neighbors):

                adj_local = adjacency_list(
                    data_raw.edge_index, 
                    data_raw.x.size(0)
                    )
                ok_lat = True
                for n in lateral:
                    for nbr in adj_local[n]:
                        if nbr == node_idx:
                            continue
                        if atomic_num_at(data_raw, nbr) != 6:
                            ok_lat = False
                            break
                    if not ok_lat:
                        break
                if ok_lat:
                    cur_double = atomic_num_at(
                        data_raw, hetero_double)
                    cur_lats = sorted([atomic_num_at(
                        data_raw, n) for n in lateral]
                        )
                    for rule in amide_family_rules():
                        left, right = rule_fragments(rule["direction"])
                        if not left or not right:
                            continue
                        if left not in left_map or right not in left_map:
                            continue
                        if (cur_double, cur_lats) != (
                            left_map[left][0],
                            sorted(left_map[left][1])):
                            continue
                        data_work = copy.copy(data_raw)
                        data_work.x = data_raw.x.clone()
                        data_work.edge_index = data_raw.edge_index.clone()
                        data_work.edge_attr = data_raw.edge_attr.clone()
                        target_double_atom, target_lats = left_map[right]
                        if atomic_num_at(data_work, 
                                hetero_double) != target_double_atom:
                            set_atomic_num(data_work,
                                hetero_double, target_double_atom
                                )
                        target_lats = list(target_lats)
                        for idx, lat in enumerate(lateral):
                            desired = target_lats[idx % len(target_lats)]
                            if atomic_num_at(data_work, lat) != desired:
                                set_atomic_num(data_work, lat, desired)
                        update_all_features(data_work)
                        normalize_edge_attr(data_work)
                        for i in range(data_work.x.size(0)):
                            atom_num = atomic_num_at(data_work, i)
                            valence, _ = node_valence_degree(data_work, i)
                            if not valence_ok(atom_num, valence):
                                break
                        else:
                            group = [node_idx, hetero_double] + list(lateral)
                            data_chk = copy.copy(data_work)
                            data_chk.x = data_work.x.clone()
                            data_chk.edge_index = data_work.edge_index.clone()
                            data_chk.edge_attr = data_work.edge_attr.clone()
                            remove_nodes(data_chk,
                                set(range(data_chk.x.size(0))) - set(group)
                                )
                            if not violates_blacklist(data_chk):
                                h = (data_work.x.cpu().numpy().tobytes(),
                                    data_work.edge_index.cpu().numpy().tobytes(),
                                    data_work.edge_attr.cpu().numpy().tobytes()
                                    )
                                if h not in seen:
                                    seen.add(h)
                                    out.append((data_work, 
                                        group, rule["rule_id"]))
    if len(out) > out_before:
        pass


def apply_carbamate_family(
    families,
    data_raw,
    node_idx,
    attachment_sp2,
    attachment_sp3,
    out,
    seen):

    if "CARBAMATE_FAMILY_ALL" not in families:
        return
    if not (attachment_sp2 or attachment_sp3):
        return

    ester_map = {
        "COOMe": {"double": 8, "lats": [8, 6],
            "reqs": ["alkoxy", None]},
        "carbonate": {"double": 8, "lats": [8, 8], 
            "reqs": ["alkoxy", "alkoxy"]},
        "carbamate": {"double": 8, "lats": [8, 7], 
            "reqs": ["alkoxy", None]},
        "thiocarbamate": {"double": 16, "lats": [8, 7], 
            "reqs": ["alkoxy", None]},
        "selenocarbamate": {"double": 34, "lats": [8, 7], 
            "reqs": ["alkoxy", None]},
        "thioester": {"double": 8, "lats": [16, 6], 
            "reqs": ["thio", None]}
        }

    def is_alkoxy_slot(carbonyl, node):
        if bond_order_between(
            data_raw, carbonyl, node) != 1.0:
            return False
        adj_local = adjacency_list(
            data_raw.edge_index, data_raw.x.size(0))
        others = [n for n in adj_local[node] if n != carbonyl]
        if len(others) == 0:
            return True
        if len(others) != 1:
            return False
        other = others[0]
        if atomic_num_at(data_raw, other) != 6:
            return False
        if bond_order_between(
            data_raw, node, other) != 1.0:
            return False
        
        return True

    def is_thio_slot(carbonyl, node):
        if bond_order_between(data_raw, carbonyl, node) != 1.0:
            return False
        adj_local = adjacency_list(
            data_raw.edge_index, data_raw.x.size(0)
            )
        others = [n for n in adj_local[node] if n != carbonyl]
        if len(others) == 0:
            return True
        if len(others) != 1:
            return False
        other = others[0]
        if atomic_num_at(data_raw, other) != 6:
            return False
        if bond_order_between(data_raw, node, other) != 1.0:
            return False
        
        return True

    def spec_matches_current(spec, 
        carbonyl, 
        hetero_double, 
        lateral):
        
        if atomic_num_at(
            data_raw, hetero_double) != spec["double"]:
            return False
        lat_nums = [atomic_num_at(
            data_raw, n) for n in lateral]
        if sorted(lat_nums) != sorted(spec["lats"]):
            return False
        reqs = spec["reqs"]
        if not any(reqs):
            return True
        perms = [lateral]
        if lateral[0] != lateral[1]:
            perms.append([lateral[1], lateral[0]])
        for perm in perms:
            ok = True
            for node, target_atom, req in zip(
                perm, spec["lats"], reqs):
                if atomic_num_at(
                    data_raw, node) != target_atom:
                    ok = False
                    break
                if req == "alkoxy" and not is_alkoxy_slot(
                    carbonyl, node):
                    ok = False
                    break
                if req == "thio" and not is_thio_slot(
                    carbonyl, node):
                    ok = False
                    break
            if ok:
                return True
        return False

    def choose_target_assignment(
            spec, carbonyl, lateral):
        perms = [lateral]
        if lateral[0] != lateral[1]:
            perms.append([lateral[1], lateral[0]])
        for perm in perms:
            ok = True
            for node, req in zip(perm, spec["reqs"]):
                if req == "alkoxy" and not is_alkoxy_slot(carbonyl, node):
                    ok = False
                    break
                if req == "thio" and not is_thio_slot(carbonyl, node):
                    ok = False
                    break
            if ok:
                return list(
                    zip(perm, spec["lats"]))
        return None

    out_before = len(out)
    adj_local = adjacency_list(
        data_raw.edge_index, data_raw.x.size(0))

    def extract_core(
        carbonyl, 
        required_lateral=None):

        src, dst = data_raw.edge_index
        orders = bond_orders(data_raw.edge_attr)
        mask = (src == carbonyl) | (dst == carbonyl)
        hetero_double = None
        lateral = []
        other_neighbors = []
        for k in torch.where(mask)[0].tolist():
            nbr = int(dst[k].item()) if int(
                src[k].item()) == carbonyl else int(src[k].item())
            order = float(orders[k].item())
            atom = atomic_num_at(data_raw, nbr)
            if order == 2.0 and atom in (7, 8, 16, 34):
                hetero_double = nbr
            elif order == 1.0:
                lateral.append(nbr)
            else:
                other_neighbors.append(nbr)
        if hetero_double is None or len(lateral) != 2:
            return None
        if required_lateral is not None and required_lateral not in lateral:
            return None
        if any(atomic_num_at(data_raw, n
            ) != 6 for n in other_neighbors):
            return None
        return hetero_double, lateral

    cores = []
    seen_cores = set()
    if atomic_num_at(data_raw, node_idx) == 6:
        core = extract_core(node_idx)
        if core:
            hetero_double, lateral = core
            key = (node_idx, 
                   hetero_double, tuple(sorted(lateral)))
            if key not in seen_cores:
                seen_cores.add(key)
                cores.append((
                    node_idx, hetero_double, lateral))
    for nbr in adj_local[node_idx]:
        if atomic_num_at(data_raw, nbr) != 6:
            continue
        if bond_order_between(
            data_raw, node_idx, nbr) != 1.0:
            continue
        core = extract_core(
            nbr, required_lateral=node_idx)
        if core:
            hetero_double, lateral = core
            key = (nbr, hetero_double, tuple(sorted(lateral)))
            if key not in seen_cores:
                seen_cores.add(key)
                cores.append((nbr, hetero_double, lateral))
    for hetero in adj_local[node_idx]:
        if atomic_num_at(
            data_raw, hetero) not in (8, 16, 34):
            continue
        if bond_order_between(
            data_raw, node_idx, hetero) != 1.0:
            continue
        for nbr in adj_local[hetero]:
            if nbr == node_idx:
                continue
            if atomic_num_at(data_raw, nbr) != 6:
                continue
            if bond_order_between(
                data_raw, hetero, nbr) != 1.0:
                continue
            core = extract_core(
                nbr, required_lateral=hetero)
            if core:
                hetero_double, lateral = core
                key = (nbr, hetero_double, tuple(sorted(lateral)))
                if key not in seen_cores:
                    seen_cores.add(key)
                    cores.append((nbr, hetero_double, lateral))

    for carbonyl, hetero_double, lateral in cores:
        for rule in carbamate_family_rules():
            left, right = rule_fragments(rule["direction"])
            if not left or not right:
                continue
            if left not in ester_map or right not in ester_map:
                continue
            if not spec_matches_current(
                ester_map[left], carbonyl, hetero_double, lateral):
                continue
            assignment = choose_target_assignment(
                ester_map[right], carbonyl, lateral
                )
            if assignment is None:
                continue
            data_work = copy.copy(data_raw)
            data_work.x = data_raw.x.clone()
            data_work.edge_index = data_raw.edge_index.clone()
            data_work.edge_attr = data_raw.edge_attr.clone()
            target_double = ester_map[right]["double"]
            if atomic_num_at(data_work, hetero_double) != target_double:
                set_atomic_num(data_work, hetero_double, target_double)
            for n_idx, target_atom in assignment:
                if atomic_num_at(data_work, n_idx) != target_atom:
                    set_atomic_num(data_work, n_idx, target_atom)
            update_all_features(data_work)
            normalize_edge_attr(data_work)
            for i in range(data_work.x.size(0)):
                atom_num = atomic_num_at(data_work, i)
                valence, _ = node_valence_degree(data_work, i)
                if not valence_ok(atom_num, valence):
                    break
            else:
                group = [carbonyl, hetero_double] + list(lateral)
                data_chk = copy.copy(data_work)
                data_chk.x = data_work.x.clone()
                data_chk.edge_index = data_work.edge_index.clone()
                data_chk.edge_attr = data_work.edge_attr.clone()
                remove_nodes(data_chk,
                    set(range(data_chk.x.size(0))) - set(group)
                    )
                if not violates_blacklist(data_chk):
                    h = (data_work.x.cpu().numpy().tobytes(),
                        data_work.edge_index.cpu().numpy().tobytes(),
                        data_work.edge_attr.cpu().numpy().tobytes()
                        )
                    if h not in seen:
                        seen.add(h)
                        out.append((data_work, group, rule["rule_id"]))
    if len(out) > out_before:
        pass


def apply_sulfure_family(
    families,
    data_raw,
    node_idx,
    out,
    seen):

    if "SULFURE_FAMILY_ALL" not in families:
        return

    def set_bond_order(data, i, j, order):
        src, dst = data.edge_index
        mask = ((src == i) & (dst == j)
            ) | ((src == j) & (dst == i))
        idxs = torch.where(mask)[0]
        if idxs.numel() == 0:
            return False
        e_idx = int(idxs[0].item())
        data.edge_attr[e_idx, :4] = 0.0
        if order == 1.0:
            data.edge_attr[e_idx, 0] = 1.0
        elif order == 2.0:
            data.edge_attr[e_idx, 1] = 1.0
        elif order == 3.0:
            data.edge_attr[e_idx, 2] = 1.0
        else:
            data.edge_attr[e_idx, 3] = 1.0
        normalize_edge_attr(data)
        return True

    adj_local = adjacency_list(
        data_raw.edge_index, data_raw.x.size(0))

    def find_core(node_idx_local):
        cores = []
        if atomic_num_at(data_raw, node_idx_local) in (16, 5):
            cores.append(node_idx_local)
        for nbr in adj_local[node_idx_local]:
            if atomic_num_at(data_raw, nbr) in (16, 5):
                cores.append(nbr)
        return list(dict.fromkeys(cores))

    def sulfur_ligands(s_idx):
        double_o = []
        double_n = []
        single_lig = []
        for nbr in adj_local[s_idx]:
            order = bond_order_between(data_raw, s_idx, nbr)
            atom = atomic_num_at(data_raw, nbr)
            if order == 2.0 and atom == 8:
                double_o.append(nbr)
            elif order == 2.0 and atom == 7:
                double_n.append(nbr)
            elif order == 1.0:
                single_lig.append(nbr)
        return double_o, double_n, single_lig

    def ligand_degree(node_idx_local, core_idx):
        return len([n for n in adj_local[
            node_idx_local] if n != core_idx])

    def has_aromatic_neighbor(node_idx_local, core_idx):
        for nbr in adj_local[node_idx_local]:
            if nbr == core_idx:
                continue
            slices = feature_slices()
            arom_idx = slices["aromatic"]
            if float(data_raw.x[nbr, arom_idx].item()) > 0.5:
                return True
        return False
    def has_aromatic_neighbor_work(
            data_work, node_idx_local, core_idx):
        adj_w = adjacency_list(
            data_work.edge_index, data_work.x.size(0)
            )
        slices = feature_slices()
        arom_idx = slices["aromatic"]
        for nbr in adj_w[node_idx_local]:
            if nbr == core_idx:
                continue
            if float(data_work.x[nbr, arom_idx].item()) > 0.5:
                return True
        return False

    def classify_sulfur(double_o, double_n, single_lig):
        double_total = len(double_o) + len(double_n)
        if double_total not in (1, 2):
            return None
        lig_types = [atomic_num_at(data_raw, n) for n in single_lig]
        if any(t not in (6, 7, 8) for t in lig_types):
            return None
        n_count = lig_types.count(7)
        o_count = lig_types.count(8)
        c_count = lig_types.count(6)
        if len(double_o) == 2 and len(double_n
                ) == 0 and n_count == 0 and o_count == 0:
            return "sulfone" if c_count == 2 else None
        if len(double_o) == 2 and len(double_n
                ) == 0 and n_count == 1 and o_count == 0:
            return "sulfonamide" if c_count == 1 else None
        if len(double_o) == 2 and len(double_n
                ) == 0 and o_count == 1 and n_count == 0:
            return "sulfonate" if c_count == 1 else None
        if len(double_o) == 1 and len(double_n
                ) == 0 and n_count == 0 and o_count == 0:
            return "sulfoxide" if c_count == 2 else None
        if len(double_o) == 0 and len(double_n
                ) == 1 and n_count == 0 and o_count == 0:
            return "sulfilimine" if c_count == 2 else None
        if len(double_o) == 1 and len(double_n
                ) == 1 and n_count == 0 and o_count == 0:
            return "sulfoximine" if c_count == 2 else None
        if len(double_o) == 0 and len(double_n
                ) == 2 and n_count == 0 and o_count == 0:
            return "sulfondiimine" if c_count == 2 else None
        if len(double_o) == 2 and len(double_n
                ) == 1 and n_count == 1 and o_count == 0:
            return "sulfonimidamide" if c_count == 1 else None
        if len(double_o) == 0 and len(double_n
                ) == 2 and n_count == 1 and o_count == 0:
            return "sulfondiimidamide" if c_count == 1 else None
        return None

    def classify_boron(b_idx):
        lig_types = [atomic_num_at(
            data_raw, n) for n in adj_local[b_idx]]
        o_count = lig_types.count(8)
        c_count = lig_types.count(6)
        if o_count >= 2 and c_count >= 1:
            return "boronate"
        return None


    def allow_sulfo(data_work):
        hetero = {7, 8, 16, 34}
        if data_work.edge_attr is None or data_work.edge_attr.numel() == 0:
            return False
        src, dst = data_work.edge_index
        orders = bond_orders(data_work.edge_attr)
        adj_local_chk = adjacency_list(
            data_work.edge_index, data_work.x.size(0)
            )

        def is_sulfonyl_core(idx):
            if atomic_num_at(data_work, idx) != 16:
                return False
            o_double = 0
            n_double = 0
            for nbr in adj_local_chk[idx]:
                if bond_order_between(data_work, idx, nbr) == 2.0:
                    if atomic_num_at(data_work, nbr) == 8:
                        o_double += 1
                    if atomic_num_at(data_work, nbr) == 7:
                        n_double += 1
            return (o_double + n_double) >= 1

        for e_idx, (i, j) in enumerate(
            zip(src.tolist(), dst.tolist())):
            ai = atomic_num_at(data_work, i)
            aj = atomic_num_at(data_work, j)
            order = float(orders[e_idx].item())
            if ai in hetero and aj in hetero and order == 1.0:
                if (ai == 16 and is_sulfonyl_core(i)) or (
                    aj == 16 and is_sulfonyl_core(j)):
                    continue
                return True
            if ai == 8 and aj == 8 and order == 1.0:
                return True
            if (ai == 7 and aj == 7) and order == 1.0:
                return True
        return False

    def pick_preferred(nodes, core, prefer_aromatic=False):
        if prefer_aromatic:
            arom = [n for n in nodes if has_aromatic_neighbor(n, core)]
            if arom:
                return arom
        connected = [n for n in nodes if ligand_degree(n, core) > 0]
        return connected if connected else nodes

    for core in find_core(node_idx):
        atom = atomic_num_at(data_raw, core)
        if atom == 16:
            double_o, double_n, single_lig = sulfur_ligands(core)
            cur = classify_sulfur(double_o, double_n, single_lig)
            if cur is None:
                continue
            for rule in sulfure_family_rules():
                left, right = rule_fragments(rule["direction"])
                if not left or not right or left != cur:
                    continue
                target = {
                    "sulfone": (2, 0, 0, 0),
                    "sulfonamide": (2, 0, 1, 0),
                    "sulfonate": (2, 0, 0, 1),
                    "sulfoxide": (1, 0, 0, 0),
                    "sulfilimine": (0, 1, 0, 0),
                    "sulfoximine": (1, 1, 0, 0),
                    "sulfondiimine": (0, 2, 0, 0),
                    "sulfonimidamide": (2, 1, 1, 0),
                    "sulfondiimidamide": (0, 2, 1, 0)
                }.get(right)
                if target is None:
                    continue
                target_d_o, target_d_n, target_s_n, target_s_o = target

                required_c_map = {
                    "sulfone": 2,
                    "sulfoxide": 2,
                    "sulfilimine": 2,
                    "sulfoximine": 2,
                    "sulfondiimine": 2,
                    "sulfonamide": 1,
                    "sulfonate": 1,
                    "sulfonimidamide": 1,
                    "sulfondiimidamide": 1
                    }
                required_c = required_c_map.get(right)
                if required_c is None:
                    continue

                total_lig = len(double_o
                    ) + len(double_n) + len(single_lig)
                if target_d_o + target_d_n > total_lig:
                    continue

                current_double = list(double_o
                    ) + list(double_n)
                current_single = list(single_lig)
                desired_double = target_d_o + target_d_n

                if desired_double > len(current_double):
                    double_nodes = current_double
                    single_nodes = current_single
                elif desired_double < len(current_double):
                    demote = current_double[desired_double:]
                    double_nodes = current_double[:desired_double]
                    single_nodes = current_single + demote
                else:
                    double_nodes = current_double
                    single_nodes = current_single
                original_group = [core] + list(double_nodes) + list(single_nodes)
                data_work = copy.copy(data_raw)
                data_work.x = data_raw.x.clone()
                data_work.edge_index = data_raw.edge_index.clone()
                data_work.edge_attr = data_raw.edge_attr.clone()
                need_remove = len(double_nodes) - desired_double

                if need_remove > 0:
                    removable = [n for n in double_nodes if ligand_degree(n, core) == 0]
                    if len(removable) < need_remove:
                        continue
                    remove_set = set(removable[:need_remove])
                    mapping = remove_nodes(data_work, remove_set)
                    if core not in mapping:
                        continue
                    core = mapping[core]
                    double_nodes = [mapping[n
                        ] for n in double_nodes if n not in remove_set]
                    single_nodes = [mapping[n
                        ] for n in single_nodes if n not in remove_set]
                required_single = target_s_n + target_s_o + required_c

                if len(single_nodes) > required_single:
                    removable_single = [
                        n for n in single_nodes
                        if atomic_num_at(data_work, n
                                         ) != 6 and ligand_degree(n, core) == 0
                        ]
                    need_remove = len(single_nodes) - required_single
                    if len(removable_single) < need_remove:
                        continue
                    remove_set = set(removable_single[:need_remove])
                    mapping = remove_nodes(data_work, remove_set)
                    if core not in mapping:
                        continue
                    core = mapping[core]
                    double_nodes = [mapping[n
                        ] for n in double_nodes if n not in remove_set]
                    single_nodes = [mapping[n
                        ] for n in single_nodes if n not in remove_set]
                elif len(single_nodes) < required_single:
                    need_add = required_single - len(single_nodes)
                    new_nodes = [make_atom_feature(7) for _ in range(need_add)]
                    new_ids = add_nodes(data_work, new_nodes)
                    for n in new_ids:
                        add_edge(data_work, core, n, order=1.0)
                    single_nodes = single_nodes + new_ids

                if len(single_nodes) != required_single:
                    continue
                existing_double_o = [
                    n for n in double_nodes 
                    if atomic_num_at(data_raw, n) == 8
                    ]
                existing_double_n = [
                    n for n in double_nodes 
                    if atomic_num_at(data_raw, n) == 7
                    ]
                double_for_o = existing_double_o[:target_d_o]
                double_for_n = existing_double_n[:target_d_n]
                remaining_double = [
                    n for n in double_nodes
                    if n not in double_for_o 
                    and n not in double_for_n
                    ]
                if len(double_for_n) < target_d_n:
                    pref_n = pick_preferred(remaining_double, 
                    core, prefer_aromatic=True
                    )
                    for n in pref_n:
                        if n in remaining_double and len(
                            double_for_n) < target_d_n:
                            double_for_n.append(n)
                    remaining_double = [
                        n for n in remaining_double 
                        if n not in double_for_n
                        ]
                for n in remaining_double:
                    if len(double_for_o) < target_d_o:
                        double_for_o.append(n)
                    elif len(double_for_n) < target_d_n:
                        double_for_n.append(n)
                if len(double_for_o) > target_d_o or len(
                    double_for_n) > target_d_n:
                    continue
                if any(ligand_degree(
                    n, core) > 0 for n in double_for_o):
                    continue
                missing_d_n = target_d_n - len(double_for_n)
                missing_d_o = target_d_o - len(double_for_o)
                if missing_d_n < 0 or missing_d_o < 0:
                    continue
                if missing_d_n + missing_d_o > 0:
                    new_nodes = []
                    for _ in range(missing_d_n):
                        new_nodes.append(make_atom_feature(7))
                    for _ in range(missing_d_o):
                        new_nodes.append(make_atom_feature(8))
                    new_ids = add_nodes(data_work, new_nodes)
                    for n in new_ids[:missing_d_n]:
                        add_edge(data_work, core, n, order=2.0)
                        double_for_n.append(n)
                    for n in new_ids[missing_d_n:]:
                        add_edge(data_work, core, n, order=2.0)
                        double_for_o.append(n)
                    double_nodes += new_ids
                if len(double_for_o) != target_d_o or len(
                    double_for_n) != target_d_n:
                    continue

                aromatic_c = [
                    n for n in single_nodes
                    if atomic_num_at(data_work, n) == 6
                    and has_aromatic_neighbor_work(data_work, n, core)
                    ]
                reserved_c = []
                if aromatic_c and required_c > 0:
                    reserved_c = [aromatic_c[0]]
                single_for_n = []
                if target_s_n > 0:
                    candidates = sorted(
                        [n for n in single_nodes if n not in reserved_c],
                        key=lambda n: (ligand_degree(n, core), n)
                        )
                    for n in candidates:
                        if n in single_nodes and len(single_for_n) < target_s_n:
                            single_for_n.append(n)
                if len(single_for_n) < target_s_n:
                    for n in single_nodes:
                        if n not in single_for_n and len(single_for_n) < target_s_n:
                            single_for_n.append(n)
                single_for_o = []
                if target_s_o > 0:
                    candidates = sorted(
                        [n for n in single_nodes if n not in reserved_c],
                        key=lambda n: (ligand_degree(n, core), n)
                        )
                    for n in candidates:
                        if n in single_for_n:
                            continue
                        if len(single_for_o) < target_s_o:
                            single_for_o.append(n)
                remaining_single = [
                    n for n in single_nodes
                    if n not in single_for_n + single_for_o
                    ]
                for n in reserved_c:
                    if n not in remaining_single:
                        remaining_single.append(n)
                if len(remaining_single) != required_c:
                    continue
                if required_c > 2:
                    continue
                group_original = [core] + list(
                    double_nodes) + list(single_nodes)
                for n in double_nodes:
                    if not set_bond_order(data_work, core, n, 2.0):
                        continue
                for n in single_nodes:
                    if not set_bond_order(data_work, core, n, 1.0):
                        continue
                for n in double_for_n:
                    if atomic_num_at(data_work, n) != 7:
                        set_atomic_num(data_work, n, 7)
                for n in double_for_o:
                    if atomic_num_at(data_work, n) != 8:
                        set_atomic_num(data_work, n, 8)
                for n in single_for_n:
                    if atomic_num_at(data_work, n) != 7:
                        set_atomic_num(data_work, n, 7)
                for n in single_for_o:
                    if atomic_num_at(data_work, n) != 8:
                        set_atomic_num(data_work, n, 8)
                for n in remaining_single:
                    if atomic_num_at(data_work, n) != 6:
                        set_atomic_num(data_work, n, 6)

                touched = [core] + double_nodes + single_nodes
                update_all_features(data_work)
                normalize_edge_attr(data_work)
                adj_chk = adjacency_list(
                    data_work.edge_index, data_work.x.size(0)
                    )
                dbl_o = 0
                dbl_n = 0
                s_n = 0
                s_o = 0
                s_c = 0

                for nbr in adj_chk[core]:
                    order = bond_order_between(data_work, core, nbr)
                    atom = atomic_num_at(data_work, nbr)
                    if order == 2.0 and atom == 8:
                        dbl_o += 1
                    elif order == 2.0 and atom == 7:
                        dbl_n += 1
                    elif order == 1.0 and atom == 7:
                        s_n += 1
                    elif order == 1.0 and atom == 8:
                        s_o += 1
                    elif order == 1.0 and atom == 6:
                        s_c += 1

                if (dbl_o != target_d_o
                    or dbl_n != target_d_n
                    or s_n != target_s_n
                    or s_o != target_s_o
                    or s_c != required_c):
                    continue

                for i in set(touched):
                    atom_num = atomic_num_at(data_work, i)
                    valence, _ = node_valence_degree(data_work, i)
                    if not valence_ok(atom_num, valence):
                        break
                else:
                    if not allow_sulfo(data_work):
                        h = (data_work.x.cpu().numpy().tobytes(),
                            data_work.edge_index.cpu().numpy().tobytes(),
                            data_work.edge_attr.cpu().numpy().tobytes()
                            )
                        if h not in seen:
                            seen.add(h)
                            group_expanded = {core}
                            for nbr in original_group:
                                if nbr == core:
                                    continue
                                if bond_order_between(
                                        data_raw, core, nbr) is None:
                                    continue
                                group_expanded.update(
                                    branch_nodes(data_raw, core, nbr))
                            group_for_map = {
                                n for n in group_expanded
                                if not (
                                    atomic_num_at(data_raw, n) == 6
                                    and is_ring_atom(data_raw, n)
                                )
                            }
                            out.append((data_work, sorted(
                                group_for_map), rule["rule_id"])
                                )
        elif atom == 5:
            cur = classify_boron(core)
            if cur != "boronate":
                continue
            continue


def apply_phosphorus_family(
    families,
    data_raw,
    node_idx,
    out,
    seen):

    if "PHOSPHORUS_FAMILY_ALL" not in families:
        return

    adj_local = adjacency_list(
        data_raw.edge_index, data_raw.x.size(0))

    def phospho_core_candidates(node_idx_local):
        cores = []
        if atomic_num_at(data_raw, node_idx_local) == 15:
            cores.append(node_idx_local)
        for nbr in adj_local[node_idx_local]:
            if atomic_num_at(data_raw, nbr) == 15:
                cores.append(nbr)
        return list(dict.fromkeys(cores))

    def phospho_ligands_simple(p_idx):
        double_lig = None
        single_lig = []
        for nbr in adj_local[p_idx]:
            order = bond_order_between(data_raw, p_idx, nbr)
            atom = atomic_num_at(data_raw, nbr)
            if order == 2.0 and atom in (8, 16):
                double_lig = nbr
            elif order == 1.0:
                single_lig.append(nbr)
        return double_lig, single_lig

    def ligand_degree(data_ref, node_idx_local, core_idx):
        return len([n for n in adj_local[
            node_idx_local] if n != core_idx])

    def violates_blacklist_allow_phospho(data_work):
        hetero = {7, 8, 15, 16, 34, 9, 17}
        if data_work.edge_attr is None or data_work.edge_attr.numel() == 0:
            return False
        src, dst = data_work.edge_index
        orders = bond_orders(data_work.edge_attr)
        adj_local_chk = adjacency_list(
            data_work.edge_index, data_work.x.size(0)
            )

        def is_phosphonyl(idx):
            if atomic_num_at(data_work, idx) != 15:
                return False
            for nbr in adj_local_chk[idx]:
                if atomic_num_at(data_work, nbr
                        ) in (8, 16) and bond_order_between(
                    data_work, idx, nbr) == 2.0:
                    return True
            return False

        for e_idx, (i, j) in enumerate(zip(src.tolist(), dst.tolist())):
            ai = atomic_num_at(data_work, i)
            aj = atomic_num_at(data_work, j)
            order = float(orders[e_idx].item())
            if ai in hetero and aj in hetero and order == 1.0:
                if (ai == 15 and is_phosphonyl(i)) or (aj == 15 and is_phosphonyl(j)):
                    continue
                return True
            if ai == 8 and aj == 8 and order == 1.0:
                return True
            if (ai == 7 and aj == 7) and order == 1.0:
                return True
        return False

    for core in phospho_core_candidates(node_idx):
        if node_idx != core:
            continue
        double_lig, single_lig = phospho_ligands_simple(core)
        if double_lig is None or len(single_lig) != 3:
            continue
        patterns = [
            ("phosphorothioate", 16, [8, 8, 8]),
            ("phosphate", 8, [8, 8, 8]),
            ("phosphorosulfate", 8, [16, 8, 8]),
            ("phosphorosulfate_thio", 16, [16, 8, 8]),
            ("phosphonate", 8, [6, 8, 8]),
            ("phosphorofluoridate", 8, [8, 8, 9]),
            ("phosphorosulfonamide", 8, [16, 7, 8]),
            ("phosphonate_fluoro", 8, [6, 8, 9]),
            ("phosphonate_thio", 8, [6, 16, 8]),
            ("phosphorocyanidate", 8, [8, "CN", 7]),
            ("phosphorodiamidate_CCN", 8, [6, 6, 7]),
            ("phosphorochloridate", 8, [8, 8, 17]),
            ("phosphonate_chloro", 8, [6, 8, 17]),
            ("phosphorodiamidate", 8, [8, 7, 7]),
            ("phosphinate_oxy", 8, [6, 6, 8]),
            ("phosphinothioate", 16, [6, 6, 16]),
            ("phosphinothioate_CC", 16, [6, 6, 6]),
            ("phosphinate", 8, [6, 6, 6]),
            ("phosphinothioate_oxy", 16, [6, 6, 8]),
            ("phosphinate_thio", 8, [6, 6, 16]),
            ]

        cur_double = atomic_num_at(data_raw, double_lig)
        lig_nodes = list(single_lig)
        lig_types = [atomic_num_at(
            data_raw, n) for n in lig_nodes]

        def is_halogen_allowed(node_idx_local):
            return ligand_degree(
                data_raw, node_idx_local, core) == 0

        def ligand_ok(node_idx_local, target):
            nbrs = [n for n in adj_local[
                node_idx_local] if n != core]
            orders = [
                bond_order_between(
                    data_raw, node_idx_local, n) or 0.0
                for n in nbrs
                ]
            deg = len(nbrs)
            max_order = max(orders) if orders else 0.0
            total_order = sum(orders)
            if target == "CN":
                if deg == 0:
                    return True
                if deg == 1:
                    n = nbrs[0]
                    return (atomic_num_at(data_raw, n) == 7
                        and abs(orders[0] - 3.0) < 1e-6
                        )
                return False
            if target in (9, 17):
                return deg == 0
            if target in (8, 16):
                return deg <= 1 and all(
                    abs(o - 1.0) < 1e-6 for o in orders)
            if target == 7:
                return deg <= 2 and max_order <= 2.0 and total_order <= 2.0
            if target == 6:
                return deg <= 2 and max_order <= 2.0 and total_order <= 2.0
            return False

        def match_and_apply(target_double, target_ligs, 
            rule_name, core_idx, double_idx, lig_nodes_src):
            perm_targets = set()
            for t0 in target_ligs:
                for t1 in target_ligs:
                    for t2 in target_ligs:
                        if sorted([t0, t1, t2], 
                            key=str) == sorted(target_ligs, key=str):
                            perm_targets.add((t0, t1, t2))
            for t0, t1, t2 in perm_targets:
                targets = [t0, t1, t2]
                ok = True
                nitrile_map = {}
                for node_idx_local, t in zip(lig_nodes_src, targets):
                    nitrile_present = False
                    for nbr in adj_local[node_idx_local]:
                        if nbr == core_idx:
                            continue
                        if atomic_num_at(data_raw, nbr
                            ) == 7 and bond_order_between(
                            data_raw, node_idx_local, nbr) == 3.0:
                            nitrile_map[node_idx_local] = nbr
                            nitrile_present = True
                            break
                    if t in (9, 17) and not (
                        is_halogen_allowed(node_idx_local) or nitrile_present):
                        ok = False
                        break
                    if not ligand_ok(node_idx_local, t):
                        if not (nitrile_present and t != "CN"):
                            ok = False
                            break
                if not ok:
                    continue
                if targets == lig_types and cur_double == target_double:
                    continue
                data_work = copy.copy(data_raw)
                data_work.x = data_raw.x.clone()
                data_work.edge_index = data_raw.edge_index.clone()
                data_work.edge_attr = data_raw.edge_attr.clone()
                if cur_double != target_double:
                    set_atomic_num(data_work, double_idx, target_double)
                new_nodes = []

                def ensure_nitrile(data_local, c_idx):
                    adj_n = adjacency_list(
                        data_local.edge_index, data_local.x.size(0))[c_idx]
                    for nbr in adj_n:
                        if nbr == core:
                            continue
                        if atomic_num_at(data_local, nbr
                            ) == 7 and bond_order_between(
                            data_local, c_idx, nbr) == 3.0:
                            return []
                        return None
                    n_idx = add_nodes(data_local, [make_atom_feature(7)])[0]
                    add_edge(data_local, c_idx, n_idx, order=3.0)
                    return [n_idx]

                lig_nodes = list(lig_nodes_src)
                for n, t in zip(lig_nodes, targets):
                    if t == "CN":
                        if atomic_num_at(data_work, n) != 6:
                            set_atomic_num(data_work, n, 6)
                        added = ensure_nitrile(data_work, n)
                        if added is None:
                            ok = False
                            break
                        new_nodes.extend(added)
                    else:
                        if atomic_num_at(data_work, n) != t:
                            set_atomic_num(data_work, n, t)
                        if n in nitrile_map:
                            removed_nodes = [nitrile_map[n]]
                            mapping = remove_nodes(
                                data_work, set(removed_nodes))
                            if core_idx not in mapping or double_idx not in mapping:
                                ok = False
                                break
                            core_idx = mapping[core_idx]
                            double_idx = mapping[double_idx]
                            lig_nodes[:] = [mapping[x] for x in lig_nodes if x in mapping]
                            new_nodes = [mapping[x] for x in new_nodes if x in mapping]
                if not ok:
                    continue
                group_original = [
                    core_idx, double_idx] + list(lig_nodes)
                update_all_features(data_work)
                normalize_edge_attr(data_work)
                group = [
                    n for n in group_original
                    if not (
                        atomic_num_at(data_raw, n) == 6
                        and is_ring_atom(data_raw, n)
                    )
                ]
                for i in group:
                    atom_num = atomic_num_at(data_work, i)
                    valence, _ = node_valence_degree(data_work, i)
                    if not valence_ok(atom_num, valence):
                        break
                else:
                    if not violates_blacklist_allow_phospho(data_work):
                        h = (data_work.x.cpu().numpy().tobytes(),
                            data_work.edge_index.cpu().numpy().tobytes(),
                            data_work.edge_attr.cpu().numpy().tobytes()
                            )
                        if h not in seen:
                            seen.add(h)
                            out.append((data_work, group, f"PHOSPHORUS_FAMILY_ALL_{rule_name}"))

        for name, dbl, ligs in patterns:
            match_and_apply(dbl, ligs, name, 
                    core, double_lig, lig_nodes
                    )


def apply_toggle_charge_family(
    families,
    data_raw,
    node_idx,
    out,
    seen):

    if "TOGGLE_CHARGE_FAMILY_ALL" not in families:
        return

    charge_rules = toggle_charge_family_rules()
    charge_rule_id = charge_rules[0]["rule_id"
                ] if charge_rules else "TOGGLE_CHARGE_FAMILY_ALL"

    def is_aromatic_node_local(data, idx):
        slices = feature_slices()
        arom_idx = slices["aromatic"]
        return bool(data.x[idx, arom_idx].item() > 0.5)

    def is_amine(data, idx):
        if atomic_num_at(data, idx) != 7:
            return False
        if is_aromatic_node_local(data, idx):
            return False
        valence, degree = node_valence_degree(data, idx)
        if degree not in (1, 2, 3):
            return False
        src, dst = data.edge_index
        orders = bond_orders(data.edge_attr)
        mask = (src == idx) | (dst == idx)
        if not mask.any():
            return False
        if any(float(o.item()) != 1.0 for o in orders[mask]):
            return False
        adj_local = adjacency_list(data.edge_index, data.x.size(0))
        for nbr in adj_local[idx]:
            if atomic_num_at(data, nbr) != 6:
                return False
            for nb2 in adj_local[nbr]:
                if nb2 == idx:
                    continue
                if bond_order_between(data, nbr, nb2
                        ) == 2.0 and atomic_num_at(data, nb2) in (8, 16, 34):
                    return False
        return valence <= 3.0

    def is_carboxylate_oxygen(data, idx):
        if atomic_num_at(data, idx) != 8:
            return False
        src, dst = data.edge_index
        orders = bond_orders(data.edge_attr)
        mask = (src == idx) | (dst == idx)
        if not mask.any():
            return False
        valence, degree = node_valence_degree(data, idx)
        if degree != 1:
            return False
        adj_local = adjacency_list(data.edge_index, data.x.size(0))
        for nbr in adj_local[idx]:
            if atomic_num_at(data, nbr) == 7:
                return False
        for e_idx in torch.where(mask)[0].tolist():
            if float(orders[e_idx].item()) != 1.0:
                continue
            nbr = int(dst[e_idx].item()) if int(
                src[e_idx].item()) == idx else int(src[e_idx].item())
            nbr_atom = atomic_num_at(data, nbr)
            if nbr_atom == 6:
                nbr_mask = (src == nbr) | (dst == nbr)
                if any(
                    float(orders[k].item()) == 2.0 and
                    atomic_num_at(data, int(dst[k].item()
                            ) if int(src[k].item()) == nbr else int(src[k].item())) == 8
                    for k in torch.where(nbr_mask)[0].tolist()):
                    return True
            if nbr_atom in (15, 16):
                return True
        return False

    def toggle_charge(data, idx):
        charge = atomic_charge_at(data, idx)
        if is_amine(data, idx):
            if charge == 0:
                set_formal_charge(data, idx, 1)
                update_all_features(data)
                return True
            if charge == 1:
                set_formal_charge(data, idx, 0)
                update_all_features(data)
                return True
        if is_carboxylate_oxygen(data, idx):
            if charge == 0:
                set_formal_charge(data, idx, -1)
                update_all_features(data)
                return True
            if charge == -1:
                set_formal_charge(data, idx, 0)
                update_all_features(data)
                return True
        return False

    def charge_group(data, idx):
        if is_amine(data, idx):
            adj_local = adjacency_list(data.edge_index, data.x.size(0))
            return sorted({idx} | set(adj_local[idx]))
        if is_carboxylate_oxygen(data, idx):
            adj_local = adjacency_list(data.edge_index, data.x.size(0))
            group = {idx}
            for nbr in adj_local[idx]:
                group.add(nbr)
                if atomic_num_at(data, nbr) == 6:
                    for nb2 in adj_local[nbr]:
                        if nb2 == idx:
                            continue
                        if atomic_num_at(data, nb2
                                ) == 8 and bond_order_between(data, nbr, nb2) == 2.0:
                            group.add(nb2)
            return sorted(group)
        return [idx]

    def guanidine_core(data, idx):
        if atomic_num_at(data, idx) != 7:
            return None
        if is_aromatic_node_local(data, idx):
            return None
        adj_local = adjacency_list(data.edge_index, data.x.size(0))
        for c in adj_local[idx]:
            if atomic_num_at(data, c) != 6:
                continue
            n_neighbors_local = [
                n for n in adj_local[c] if atomic_num_at(data, n) == 7
                ]
            if len(n_neighbors_local) < 3:
                continue
            if not any(
                bond_order_between(data, c, n
                        ) == 2.0 for n in n_neighbors_local):
                continue
            return c, n_neighbors_local
        return None

    g = guanidine_core(data_raw, node_idx)
    handled = False
    if g is not None:
        core_c, n_list = g
        for n in n_list:
            data_work = copy.deepcopy(data_raw)
            charge_n = atomic_charge_at(data_work, n)
            if charge_n == 0:
                set_formal_charge(data_work, n, 1)
                update_all_features(data_work)
            elif charge_n == 1:
                set_formal_charge(data_work, n, 0)
                update_all_features(data_work)
            else:
                continue
            if True:
                h = (data_work.x.cpu().numpy().tobytes(),
                    data_work.edge_index.cpu().numpy().tobytes(),
                    data_work.edge_attr.cpu().numpy().tobytes()
                    )
                if h in seen:
                    continue
                seen.add(h)
                group = sorted({core_c} | set(n_list))
                out.append((data_work, group, charge_rule_id))
                handled = True
    if not handled:
        data_work = copy.deepcopy(data_raw)
        if toggle_charge(data_work, node_idx):
            h = (data_work.x.cpu().numpy().tobytes(),
                data_work.edge_index.cpu().numpy().tobytes(),
                data_work.edge_attr.cpu().numpy().tobytes()
                )
            if h not in seen:
                seen.add(h)
                group = charge_group(data_raw, node_idx)
                out.append((data_work, group, charge_rule_id))
                

def apply_polyvalent_family(
    families,
    data_raw,
    node_idx,
    out,
    seen):

    if "POLYVALENT_FAMILY_ALL" not in families:
        return

    poly_rules = polyvalent_family_rules()
    poly_rule_id = poly_rules[0][
        "rule_id"] if poly_rules else "POLYVALENT_FAMILY_ALL_SWAP"

    def neighbors_all_c(data, idx, require_single=True):
        adj_local = adjacency_list(data.edge_index, data.x.size(0))
        for nbr in adj_local[idx]:
            if atomic_num_at(data, nbr) != 6:
                return False
            if require_single and bond_order_between(data, idx, nbr) != 1.0:
                return False
        return True

    def has_any_non_c_neighbor(data, idx):
        adj_local = adjacency_list(
            data.edge_index, data.x.size(0))
        return any(atomic_num_at(
            data, n) != 6 for n in adj_local[idx])

    def has_hetero_or_halogen_neighbor(data, idx):
        adj_local = adjacency_list(
            data.edge_index, data.x.size(0))
        return any(atomic_num_at(data, n) in (
            7, 8, 16, 34, 9, 17, 35, 53, 33, 15) for n in adj_local[idx])

    def apply_center_swap(center, target_atom, target_charge):
        data_work = copy.copy(data_raw)
        data_work.x = data_raw.x.clone()
        data_work.edge_index = data_raw.edge_index.clone()
        data_work.edge_attr = data_raw.edge_attr.clone()
        set_atomic_num(data_work, center, target_atom)
        set_formal_charge(data_work, center, target_charge)
        update_all_features(data_work)
        normalize_edge_attr(data_work)
        atom_num_new = atomic_num_at(data_work, center)
        valence, _ = node_valence_degree(data_work, center)
        if not valence_ok(atom_num_new, valence):
            if not (atom_num_new == 7 
                    and target_charge == 1 and valence <= 4.0):
                return
        else:
            pass
        h = (data_work.x.cpu().numpy().tobytes(),
            data_work.edge_index.cpu().numpy().tobytes(),
            data_work.edge_attr.cpu().numpy().tobytes()
            )
        if h in seen:
            return
        seen.add(h)
        adj_local = adjacency_list(
            data_raw.edge_index, data_raw.x.size(0))
        group = {center}
        slices = feature_slices()
        arom_idx = slices["aromatic"]
        for nbr in adj_local[center]:
            if atomic_num_at(data_raw, center) != 5:
                if bool(data_raw.x[nbr, arom_idx].item() > 0.5):
                    continue
                if is_ring_atom(data_raw, nbr):
                    continue
            group.add(nbr)
        group = sorted(group)
        out.append((data_work, group, poly_rule_id))

    src, dst = data_raw.edge_index
    orders = bond_orders(data_raw.edge_attr)
    for e_idx in torch.where((src == node_idx
            ) | (dst == node_idx))[0].tolist():
        if float(orders[e_idx].item()) != 2.0:
            continue
        i = int(src[e_idx].item())
        j = int(dst[e_idx].item())
        if atomic_num_at(data_raw, i
            ) != 33 or atomic_num_at(data_raw, j) != 33:
            continue
        if i > j:
            i, j = j, i
        for ti, tj in ((7, 7), (6, 6)):
            data_work = copy.copy(data_raw)
            data_work.x = data_raw.x.clone()
            data_work.edge_index = data_raw.edge_index.clone()
            data_work.edge_attr = data_raw.edge_attr.clone()
            set_atomic_num(data_work, i, ti)
            set_atomic_num(data_work, j, tj)
            update_all_features(data_work)
            normalize_edge_attr(data_work)
            for n in (i, j):
                atom_num_new = atomic_num_at(data_work, n)
                valence, _ = node_valence_degree(data_work, n)
                if not valence_ok(atom_num_new, valence):
                    break
            else:
                h = (data_work.x.cpu().numpy().tobytes(),
                    data_work.edge_index.cpu().numpy().tobytes(),
                    data_work.edge_attr.cpu().numpy().tobytes()
                    )
                if h in seen:
                    continue
                seen.add(h)
                adj_local = adjacency_list(
                    data_raw.edge_index, data_raw.x.size(0))
                group = sorted({i, j} | set(adj_local[i]) | set(adj_local[j]))
                out.append((data_work, group, poly_rule_id))

    atom_num = atomic_num_at(data_raw, node_idx)
    valence, degree = node_valence_degree(data_raw, node_idx)

    if degree == 4 and neighbors_all_c(
        data_raw, node_idx, require_single=True):
        if atom_num in (15, 7, 33, 14):
            for target in (15, 7, 33, 14, 6):
                if target == atom_num:
                    continue
                charge_val = 0
                if target == 7:
                    charge_val = 1
                apply_center_swap(node_idx, target, charge_val)

    if atom_num in (33, 15, 14) and degree == 3 and neighbors_all_c(
        data_raw, node_idx, require_single=True):
        if has_hetero_or_halogen_neighbor(data_raw, node_idx):
            apply_center_swap(node_idx, 6, 0)
        else:
            apply_center_swap(node_idx, 7, 0)
            apply_center_swap(node_idx, 14, 0)
            apply_center_swap(node_idx, 6, 0)

    if atom_num == 33 and degree == 2:
        adj_local = adjacency_list(
            data_raw.edge_index, data_raw.x.size(0))
        orders_local = [bond_order_between(
            data_raw, node_idx, n) for n in adj_local[node_idx]]
        if 2.0 in orders_local and 1.0 in orders_local:
            apply_center_swap(node_idx, 6, 0)
        elif all(o == 1.0 for o in orders_local):
        
            if not has_any_non_c_neighbor(data_raw, node_idx):
                apply_center_swap(node_idx, 7, 0)
            apply_center_swap(node_idx, 6, 0)
    if atom_num == 5:
        apply_center_swap(node_idx, 6, 0)
        apply_center_swap(node_idx, 14, 0)
    if atom_num == 14:
        apply_center_swap(node_idx, 5, 0)
        apply_center_swap(node_idx, 6, 0)


def apply_aliphatic_family(
    families,
    data_raw,
    node_idx,
    out,
    seen):

    if "ALIPHATIC_FAMILY_ALL" not in families:
        return

    aliphatic_rules = aliphatic_family_rules()
    aliphatic_rule_id = aliphatic_rules[0]["rule_id"
            ] if aliphatic_rules else "ALIPHATIC_FAMILY_ALL_ISOSTERIC"

    def is_aromatic_node_local(data, n):
        slices = feature_slices()
        arom_idx = slices["aromatic"]
        return bool(data.x[n, arom_idx].item() > 0.5)

    def is_ring_atom_local(data, idx):
        if data.edge_attr is None or data.edge_attr.numel() == 0:
            return False
        if data.edge_attr.size(1) < 6:
            return False
        src, dst = data.edge_index
        mask = (src == idx) | (dst == idx)
        if not mask.any():
            return False
        return bool((data.edge_attr[mask, 5] > 0.5).any().item())

    def is_ring_substituent(data, idx):
        if is_ring_atom_local(data, idx) or is_aromatic_node_local(data, idx):
            return False
        adj_local = adjacency_list(data.edge_index, data.x.size(0))
        for nbr in adj_local[idx]:
            if is_ring_atom_local(data, nbr) or is_aromatic_node_local(data, nbr):
                return True
        return False

    def is_linear_aliphatic_carbon(data, idx):
        if atomic_num_at(data, idx) != 6:
            return False
        if is_aromatic_node_local(data, idx) or is_ring_atom_local(data, idx):
            return False
        if is_ring_substituent(data, idx):
            return False
        adj_local = adjacency_list(data.edge_index, data.x.size(0))
        carbon_neighbors = []
        for nbr in adj_local[idx]:
            if atomic_num_at(data, nbr) != 6:
                return False
            if is_aromatic_node_local(data, nbr) or is_ring_atom_local(data, nbr):
                return False
            if bond_order_between(data, idx, nbr) != 1.0:
                return False
            carbon_neighbors.append(nbr)
        return len(carbon_neighbors) == 2

    if is_linear_aliphatic_carbon(data_raw, node_idx):
        if not adjacent_to_carbonyl(data_raw, node_idx):
            atom_num = atomic_num_at(data_raw, node_idx)
            for target in (6, 7, 8, 16):
                if target == atom_num:
                    continue
                data_work = copy.copy(data_raw)
                data_work.x = data_raw.x.clone()
                data_work.edge_index = data_raw.edge_index.clone()
                data_work.edge_attr = data_raw.edge_attr.clone()
                set_atomic_num(data_work, node_idx, target)
                update_all_features(data_work)
                normalize_edge_attr(data_work)
                atom_num_new = atomic_num_at(data_work, node_idx)
                valence, _ = node_valence_degree(data_work, node_idx)
                if not valence_ok(atom_num_new, valence):
                    continue
                h = (
                    data_work.x.cpu().numpy().tobytes(),
                    data_work.edge_index.cpu().numpy().tobytes(),
                    data_work.edge_attr.cpu().numpy().tobytes()
                )
                if h in seen:
                    continue
                seen.add(h)
                out.append((data_work, [node_idx], aliphatic_rule_id))
    return out


def apply_toggle_ring_family(
    families,
    data_raw,
    node_idx,
    out,
    seen):

    if "TOGGLE_RING_FAMILY_ALL" not in families:
        return

    toggle_rules = toggle_ring_family_rules()
    rule_id_toggle = toggle_rules[0][
        "rule_id"] if toggle_rules else "TOGGLE_AROMATIC"

    def ring_has_highvalent_ps(data, nodes):
        for n in nodes:
            if atomic_num_at(data, n) not in (15, 16):
                continue
            if not is_ring_atom(data, n):
                continue
            valence, _ = node_valence_degree(data, n)
            if valence >= 4.0:
                return True
        return False

    def is_ring_atom_local(data, idx):
        slices = feature_slices()
        arom_idx = slices["aromatic"]
        if float(data.x[idx, arom_idx].item()) > 0.5:
            return True
        if data.edge_attr is None or data.edge_attr.numel() == 0:
            pass
        if data.edge_attr.numel() > 0 and data.edge_attr.size(1) >= 6:
            src, dst = data.edge_index
            mask = (src == idx) | (dst == idx)
            if mask.any() and bool((data.edge_attr[mask, 5] > 0.5).any().item()):
                return True
        G = nx.Graph()
        G.add_nodes_from(range(data.x.size(0)))
        G.add_edges_from(data.edge_index.t().tolist())
        for cycle in nx.cycle_basis(G):
            if len(cycle) in (5, 6) and idx in cycle:

                return True
        return False

    def ring_component(data, idx):
        if not is_ring_atom_local(data, idx):
            return set(), []
        if data.edge_attr is None or data.edge_attr.numel() == 0:
            return set(), []
        src, dst = data.edge_index
        ring_edges = []
        if data.edge_attr.size(1) >= 6:
            for e_idx in range(data.edge_attr.size(0)):
                if data.edge_attr[e_idx, 5].item() > 0.5:
                    ring_edges.append(e_idx)
        if ring_edges:
            G = nx.Graph()
            G.add_nodes_from(range(data.x.size(0)))
            for e_idx in ring_edges:
                i = int(src[e_idx].item())
                j = int(dst[e_idx].item())
                G.add_edge(i, j, e_idx=e_idx)
            best = None
            for cycle in nx.cycle_basis(G):
                if len(cycle) not in (5, 6):
                    continue
                if idx not in cycle:
                    continue
                if best is None or len(cycle) < len(best):
                    best = cycle
            if best:
                nodes = set(best)
                edges = []
                for u, v in zip(best, best[1:] + [best[0]]):
                    e_idx = G.edges[u, v].get("e_idx")
                    if e_idx is not None:
                        edges.append(e_idx)
                return nodes, edges
        G = nx.Graph()
        G.add_nodes_from(range(data.x.size(0)))
        G.add_edges_from(data.edge_index.t().tolist())
        best = None
        for cycle in nx.cycle_basis(G):
            if len(cycle) not in (5, 6):
                continue
            if idx not in cycle:
                continue
            if best is None or len(cycle) < len(best):
                best = cycle
        if not best:
            return set(), []
        nodes = set(best)
        edges = []
        for u, v in zip(best, best[1:] + [best[0]]):
            edge_idx = None
            for k in range(data.edge_index.size(1)):
                a = int(src[k].item())
                b = int(dst[k].item())
                if (a == u and b == v) or (a == v and b == u):
                    edge_idx = k
                    break
            if edge_idx is not None:
                edges.append(edge_idx)
        return nodes, edges

    def set_edge_order(data, e_idx, order):
        data.edge_attr[e_idx, :4] = 0.0
        if order == 1.0:
            data.edge_attr[e_idx, 0] = 1.0
        elif order == 2.0:
            data.edge_attr[e_idx, 1] = 1.0
        elif order == 3.0:
            data.edge_attr[e_idx, 2] = 1.0
        else:
            data.edge_attr[e_idx, 3] = 1.0

    adj_local = adjacency_list(
        data_raw.edge_index, data_raw.x.size(0))

    def toggle_ring_aromatic(data, idx):
        nodes, edges = ring_component(data, idx)
        if not nodes or not edges:
            return None, None
        if ring_has_highvalent_ps(data, nodes):
            return None, None
        ring_has_b = any(atomic_num_at(
            data, n) == 5 for n in nodes)
        if ring_has_b:
            return None, None
        ring_size = len(nodes)
        if ring_size not in (5, 6):
            return None, None
        orders = bond_orders(data.edge_attr)
        aromatic = any(abs(float(orders[e].item()
            ) - 1.5) < 1e-6 for e in edges)
        if aromatic:
            for e_idx in edges:
                set_edge_order(data, e_idx, 1.0)
            for n in nodes:
                set_aromatic(data, n, False)
                set_hybridization(data, n, HYBRIDIZATION_CATEGORIES[2])
                set_degree(data, n, node_valence_degree(data, n)[1])
            update_all_features(data)
            normalize_edge_attr(data)
            return data, nodes
        if ring_size == 5 and all(
            atomic_num_at(data, n) == 6 for n in nodes):
            return None, None
        if not aromatic and ring_size >= 6:
            if any(atomic_num_at(data, n) in (8, 16) for n in nodes):
                return None, None
        for n in nodes:
            if atomic_num_at(data, n) != 6:
                continue
            for nbr in adj_local[n]:
                if bond_order_between(data, n, nbr
                ) == 2.0 and atomic_num_at(data, nbr) in (8, 16, 34):
                    return None, None
        if not aromatic:
            G = nx.Graph()
            G.add_nodes_from(range(data.x.size(0)))
            G.add_edges_from(data.edge_index.t().tolist())
            cycle_hits = 0
            for cycle in nx.cycle_basis(G):
                if any(n in nodes for n in cycle):
                    cycle_hits += 1
            if cycle_hits >= 2:
                return None, None

        for e_idx in edges:
            set_edge_order(data, e_idx, 1.5)
        for n in nodes:
            set_aromatic(data, n, True)
            set_hybridization(data, n, HYBRIDIZATION_CATEGORIES[1])
            set_degree(data, n, node_valence_degree(data, n)[1])
        update_all_features(data)
        normalize_edge_attr(data)
        return data, nodes

    data_work = copy.deepcopy(data_raw)
    toggled, group = toggle_ring_aromatic(data_work, node_idx)
    if toggled is not None and group:
        h = (toggled.x.cpu().numpy().tobytes(),
            toggled.edge_index.cpu().numpy().tobytes(),
            toggled.edge_attr.cpu().numpy().tobytes()
            )
        if h not in seen:
            seen.add(h)
            out.append((toggled, sorted(group), rule_id_toggle))


def apply_ring_family(
    families,
    data_raw,
    node_idx,
    out,
    seen):

    if "RING_FAMILY_ALL" not in families:
        return

    ring_rules = ring_family_rules()
    ring_rule_id = ring_rules[0]["rule_id"
        ] if ring_rules else "RING_FAMILY_ALL_ISOSTERIC"

    def ring_component_any(data, idx):
        n_nodes = data.x.size(0)
        if idx >= n_nodes:
            return set()
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        G.add_edges_from(data.edge_index.t().tolist())
        cycles = nx.cycle_basis(G)
        best = None
        for cycle in cycles:
            if idx not in cycle:
                continue
            if best is None or len(cycle) < len(best):
                best = cycle
        return set(best) if best else set()

    def ring_has_highvalent_ps(data, nodes):
        for n in nodes:
            if atomic_num_at(data, n) not in (15, 16):
                continue
            if not is_ring_atom(data, n):
                continue
            valence, _ = node_valence_degree(data, n)
            if valence >= 4.0:
                return True
        return False

    ring_nodes = ring_component_any(data_raw, node_idx)
    if ring_nodes:
        def ring_has_carbonyl_c(data, nodes):
            for n in nodes:
                if atomic_num_at(data, n) != 6:
                    continue
                if not is_ring_atom(data, n):
                    continue
                for nbr in adjacency_list(
                        data.edge_index, data.x.size(0))[n]:
                    if bond_order_between(data, n, nbr) == 2.0 and atomic_num_at(
                            data, nbr) in (8, 16, 34):
                        return True
            return False
        if ring_has_carbonyl_c(data_raw, ring_nodes):
            ring_nodes = set()
        if ring_has_highvalent_ps(data_raw, ring_nodes):
            ring_nodes = set()
        ring_has_b = any(atomic_num_at(
            data_raw, n) == 5 for n in ring_nodes)
        if ring_has_b:
            ring_nodes = set()
        slices = feature_slices()
        arom_idx = slices["aromatic"]
        is_aromatic_ring = any(
            bool(data_raw.x[n, arom_idx].item(
            ) > 0.5) for n in ring_nodes
        )
        atom_num = atomic_num_at(data_raw, node_idx)
        if atom_num not in (6, 7, 8, 16):
            ring_nodes = set()
    if ring_nodes:
        degree = node_valence_degree(data_raw, node_idx)[1]
        neighbors = adjacency_list(
            data_raw.edge_index, data_raw.x.size(0))[node_idx]
        hetero_neighbors = any(
            atomic_num_at(data_raw, n) in (7, 8, 16) for n in neighbors
            )
        allowed = [6, 7, 8, 16]
        ring_size = len(ring_nodes)
        if is_aromatic_ring:
            if ring_size == 5:
                if atom_num == 6:
                    allowed = [7]
                else:
                    allowed = [7, 8, 16]
            else:
                allowed = [6, 7]
        for target in allowed:
            if target == atom_num:
                continue
            if is_aromatic_ring and ring_size == 5:
                if atom_num == 7 and degree > 2 and target in (8, 16):
                    continue
            if not is_aromatic_ring and target in (7, 8, 16) and hetero_neighbors:
                continue
            if target in (8, 16) and degree > 2:
                continue
            data_work = copy.copy(data_raw)
            data_work.x = data_raw.x.clone()
            data_work.edge_index = data_raw.edge_index.clone()
            data_work.edge_attr = data_raw.edge_attr.clone()
            set_atomic_num(data_work, node_idx, target)
            update_all_features(data_work)
            normalize_edge_attr(data_work)
            atom_num_new = atomic_num_at(data_work, node_idx)
            valence, _ = node_valence_degree(data_work, node_idx)
            if not valence_ok(atom_num_new, valence):
                continue
            if is_aromatic_ring and target in (8, 16):
                continue
            h = (data_work.x.cpu().numpy().tobytes(),
                data_work.edge_index.cpu().numpy().tobytes(),
                data_work.edge_attr.cpu().numpy().tobytes()
                )
            if h in seen:
                continue
            seen.add(h)
            out.append(
                (data_work, sorted(ring_nodes), f"{ring_rule_id}_{atom_num}_TO_{target}")
            )


def apply_bond_family(
    families,
    data_raw,
    node_idx,
    out,
    seen,
    set_bond_order):

    if "BOND_FAMILY_ALL" not in families:
        return

    bond_rules = bond_family_rules()
    bond_rule_id = bond_rules[0]["rule_id"
                ] if bond_rules else "BOND_FAMILY_ALL_PERTURB"

    def bond_in_ring(data, edge_idx):
        if data.edge_attr is None or data.edge_attr.numel() == 0:
            return False
        if data.edge_attr.size(1) >= 6:
            return bool(data.edge_attr[edge_idx, 5].item() > 0.5)
        return False

    def implicit_h(atom_num, valence):
        return max(MAX_VALENCE.get(atom_num, 4) - valence, 0.0)

    def is_aromatic_node(data, n):
        slices = feature_slices()
        arom_idx = slices["aromatic"]
        return bool(data.x[n, arom_idx].item() > 0.5)

    src, dst = data_raw.edge_index
    orders = bond_orders(data_raw.edge_attr)
    incident = torch.where((src == node_idx) | (dst == node_idx))[0].tolist()
    order_vals = [1.0, 2.0, 3.0]
    for e_idx in incident:
        if bond_in_ring(data_raw, e_idx):
            continue
        i = int(src[e_idx].item())
        j = int(dst[e_idx].item())
        ai = atomic_num_at(data_raw, i)
        aj = atomic_num_at(data_raw, j)
        if is_aromatic_node(data_raw, i) or is_aromatic_node(data_raw, j):
            continue
        if atomic_charge_at(data_raw, i) != 0 or atomic_charge_at(data_raw, j) != 0:
            continue
  
        if not ((ai == 6 and aj == 6) or (
                ai == 6 and aj == 7) or (ai == 7 and aj == 6)):
            continue
 
        if ai == 6 and adjacent_to_carbonyl(data_raw, i):
            continue
        if aj == 6 and adjacent_to_carbonyl(data_raw, j):
            continue

        if ai == 7 and adjacent_to_carbonyl(data_raw, i):
            continue
        if aj == 7 and adjacent_to_carbonyl(data_raw, j):
            continue
        old_order = float(orders[e_idx].item())
        candidates = [o for o in order_vals if o != old_order]
        for new_order in candidates:

            def count_multiple(atom_idx, new_order_local, old_order_local):
                mask = (src == atom_idx) | (dst == atom_idx)
                if not mask.any():
                    return 0
                vals = orders[mask].tolist()
                count = sum(1 for o in vals if o >= 2.0)
                if atom_idx in (i, j):
                    if old_order_local >= 2.0:
                        count -= 1
                    if new_order_local >= 2.0:
                        count += 1
                return count

            if count_multiple(i, new_order, old_order) > 1:
                continue
            if count_multiple(j, new_order, old_order) > 1:
                continue
            val_i, _ = node_valence_degree(data_raw, i)
            val_j, _ = node_valence_degree(data_raw, j)
            atom_i = atomic_num_at(data_raw, i)
            atom_j = atomic_num_at(data_raw, j)
            new_val_i = val_i - old_order + new_order
            new_val_j = val_j - old_order + new_order
            if (new_val_i > MAX_VALENCE.get(atom_i, 4)
                    or new_val_j > MAX_VALENCE.get(atom_j, 4)
                    or implicit_h(atom_i, new_val_i) < 0
                    or implicit_h(atom_j, new_val_j) < 0):
                continue
            data_work = copy.deepcopy(data_raw)
            if not set_bond_order(data_work, i, j, new_order):
                continue
            update_all_features(data_work)
            normalize_edge_attr(data_work)
            h = (
                data_work.x.cpu().numpy().tobytes(),
                data_work.edge_index.cpu().numpy().tobytes(),
                data_work.edge_attr.cpu().numpy().tobytes()
            )
            if h in seen:
                continue
            seen.add(h)
            out.append((data_work, sorted({i, j}), bond_rule_id))


def apply_diaryl_family(
    families,
    data_raw,
    node_idx,
    out,
    seen):

    if "DIARYL_FAMILY_ALL" not in families:
        return

    diatomic_rules = diaryl_family_rules()
    diatomic_rule_id = diatomic_rules[0]["rule_id"
                        ] if diatomic_rules else "DIARYL_FAMILY_ALL_SWAP"

    def bond_in_ring(data, edge_idx):
        if data.edge_attr is None or data.edge_attr.numel() == 0:
            return False
        if data.edge_attr.size(1) >= 6:
            return bool(data.edge_attr[edge_idx, 5].item() > 0.5)
        return False

    swap_map = {
        (7, 7): [(6, 6)],  # N-N -> C-C
        (8, 8): [(6, 6)],  # O-O -> C-C (one-way)
        (16, 16): [(6, 6)],  # S-S -> C-C (one-way)
        (33, 33): [(7, 7), (6, 6)],  # As-As -> N-N or C-C (one-way)
        (7, 8): [(6, 6)]  # N-O -> C-C (one-way)
        }

    def n_neighbors(data, n, atom_num=None, orders=(1.0, 2.0)):
        adj = adjacency_list(data.edge_index, data.x.size(0))
        out = []
        for nbr in adj[n]:
            if atom_num is not None and atomic_num_at(data, nbr) != atom_num:
                continue
            if bond_order_between(
                data, n, nbr) in orders:
                out.append(nbr)
        return out

    def find_acyl_carbonyl(data, n1, n2):
        adj = adjacency_list(data.edge_index, data.x.size(0))
        for n in (n1, n2):
            for c in n_neighbors(
                data, n, atom_num=6, orders=(1.0,)):
                o_atom = None
                r_atom = None
                for nb2 in adj[c]:
                    if nb2 == n:
                        continue
                    if atomic_num_at(data, nb2
                                     ) == 8 and bond_order_between(
                                        data, c, nb2) == 2.0:
                        o_atom = nb2
                    else:
                        r_atom = nb2
                if o_atom is not None and r_atom is not None:
                    return c, o_atom, n, r_atom
        return None, None, None, None

    def hydrazone_info(data, n1, n2):
        imine_c = None
        for n in (n1, n2):
            for c in n_neighbors(
                data, n, atom_num=6, orders=(2.0,)):
                imine_c = c
        return imine_c

    def oxime_info(data, n1, n2):
        adj = adjacency_list(
            data.edge_index, data.x.size(0))
        for n in (n1, n2):
            if atomic_num_at(data, n) != 7:
                continue
            for c in n_neighbors(
                data, n, atom_num=6, orders=(2.0,)):
                if any(atomic_num_at(data, nbr) == 8 and
                    bond_order_between(data, n, nbr) == 1.0
                    for nbr in adj[n]):
                    return c
        return None

    def hydroxamic_info(data, n1, n2):
        c, o, n_attached, r_atom = find_acyl_carbonyl(data, n1, n2)
        if c is None:
            return None, None, None, None
        n = n_attached
        adj = adjacency_list(data.edge_index, data.x.size(0))
        if any(
            atomic_num_at(data, nbr) == 8 and
            bond_order_between(data, n, nbr) == 1.0
            for nbr in adj[n]):
            return c, o, n, r_atom
        return None, None, None, None

    def hydrazine_group(data, n1, n2):
        group = {n1, n2}
        carbonyl_c, carbonyl_o, _, r_atom = find_acyl_carbonyl(data, n1, n2)
        if carbonyl_c is not None:
            group.update({carbonyl_c, carbonyl_o, r_atom})
        imine_c = hydrazone_info(data, n1, n2)
        if imine_c is not None:
            group.add(imine_c)
        return sorted(group)

    def hydroxamic_or_oxime_group(data, n1, n2):
        group = {n1, n2}
        carbonyl_c, carbonyl_o, _, r_atom = hydroxamic_info(data, n1, n2)
        if carbonyl_c is not None:
            group.update({carbonyl_c, carbonyl_o, r_atom})
        imine_c = oxime_info(data, n1, n2)
        if imine_c is not None:
            group.add(imine_c)
        return sorted(group)

    def deacylate(data_in, n1, n2, swap_to=None):
        carbonyl_c, carbonyl_o, n_attached, r_atom = find_acyl_carbonyl(
            data_in, n1, n2
            )
        if carbonyl_c is None or carbonyl_o is None or r_atom is None:
            return None, None
        data_work = copy.deepcopy(data_in)
        if swap_to is not None:
            set_atomic_num(data_work, n1, swap_to[0])
            set_atomic_num(data_work, n2, swap_to[1])
        remove_set = {carbonyl_c, carbonyl_o}
        mapping = remove_nodes(data_work, remove_set)
        if n_attached not in mapping or r_atom not in mapping:
            return None, None
        n_attached_m = mapping[n_attached]
        r_atom_m = mapping[r_atom]
        if bond_order_between(
            data_work, 
            n_attached_m, 
            r_atom_m) is None:
            add_edge(data_work, 
                n_attached_m, r_atom_m, order=1.0)
        update_all_features(data_work)
        normalize_edge_attr(data_work)
        for n in (n_attached_m, r_atom_m):
            atom_num = atomic_num_at(data_work, n)
            valence, _ = node_valence_degree(data_work, n)
            if not valence_ok(atom_num, valence):
                return None, None
        group_nodes = sorted({n1, n2, carbonyl_c, carbonyl_o, r_atom})
        return data_work, group_nodes

    def filter_ring_carbons(data, nodes):
        filtered = []
        for n in nodes:
            if atomic_num_at(data, n) == 6 and is_ring_atom(data, n):
                continue
            filtered.append(n)
        return sorted(set(filtered))

    src, dst = data_raw.edge_index
    orders = bond_orders(data_raw.edge_attr)
    incident = torch.where(
        (src == node_idx) | (dst == node_idx))[0].tolist()
    for e_idx in incident:
        if bond_in_ring(data_raw, e_idx):
            continue
        if float(orders[e_idx].item()) != 1.0:
            continue
        i = int(src[e_idx].item())
        j = int(dst[e_idx].item())
        ai = atomic_num_at(data_raw, i)
        aj = atomic_num_at(data_raw, j)
        if ai > aj:
            ai, aj = aj, ai
            i, j = j, i
        key = (ai, aj)
        if key not in swap_map:
            continue
        slices = feature_slices()
        arom_idx = slices["aromatic"]
        if bool(data_raw.x[i, arom_idx].item() > 0.5
            ) or bool(data_raw.x[j, arom_idx].item() > 0.5):
            continue
        if atomic_charge_at(data_raw, i
            ) != 0 or atomic_charge_at(data_raw, j) != 0:
            continue
        extra_group = None
        is_acyl = False
        swap_targets = list(swap_map[key])
        if key == (7, 7):
            carbonyl_c, _, _, _ = find_acyl_carbonyl(data_raw, i, j)
            if carbonyl_c is not None:
                is_acyl = True
                extra_group = hydrazine_group(data_raw, i, j)
                if (7, 8) not in swap_targets:
                    swap_targets.append((7, 8))
                deacyl_data, deacyl_group = deacylate(
                    data_raw, i, j, swap_to=None
                    )
                if deacyl_data is not None:
                    h = (
                        deacyl_data.x.cpu().numpy().tobytes(),
                        deacyl_data.edge_index.cpu().numpy().tobytes(),
                        deacyl_data.edge_attr.cpu().numpy().tobytes()
                    )
                    if h not in seen:
                        seen.add(h)
                        deacyl_group = filter_ring_carbons(
                            data_raw, deacyl_group
                            )
                        out.append((deacyl_data,
                                deacyl_group, "DIARYL_FAMILY_ALL_DEACYLATE"))
            if hydrazone_info(data_raw, i, j) is not None:
                if (7, 8) not in swap_targets:
                    swap_targets.append((7, 8))
        if key == (7, 8):
            extra_group = hydroxamic_or_oxime_group(data_raw, i, j)
            carbonyl_c, _, _, _ = hydroxamic_info(data_raw, i, j)
            if carbonyl_c is not None:
                if (7, 7) not in swap_targets:
                    swap_targets.append((7, 7))
            if oxime_info(data_raw, i, j) is not None:
                if (7, 7) not in swap_targets:
                    swap_targets.append((7, 7))
        for ti, tj in swap_targets:
            data_work = copy.deepcopy(data_raw)
            set_atomic_num(data_work, i, ti)
            set_atomic_num(data_work, j, tj)
            update_all_features(data_work)
            normalize_edge_attr(data_work)
            for n in (i, j):
                atom_num = atomic_num_at(data_work, n)
                valence, _ = node_valence_degree(data_work, n)
                if not valence_ok(atom_num, valence):
                    break
            else:
                h = (
                    data_work.x.cpu().numpy().tobytes(),
                    data_work.edge_index.cpu().numpy().tobytes(),
                    data_work.edge_attr.cpu().numpy().tobytes()
                )
                if h in seen:
                    continue
                seen.add(h)
                group = extra_group if extra_group else sorted({i, j})
                group = filter_ring_carbons(data_raw, group)
                out.append((data_work, group, diatomic_rule_id))
                if is_acyl:
                    deacyl_data, deacyl_group = deacylate(
                        data_raw, i, j, swap_to=(ti, tj)
                    )
                    if deacyl_data is not None:
                        h2 = (deacyl_data.x.cpu().numpy().tobytes(),
                            deacyl_data.edge_index.cpu().numpy().tobytes(),
                            deacyl_data.edge_attr.cpu().numpy().tobytes()
                            )
                        if h2 not in seen:
                            seen.add(h2)
                            deacyl_group = filter_ring_carbons(
                                data_raw, deacyl_group
                                )
                            out.append((deacyl_data,
                                    deacyl_group, "DIARYL_FAMILY_ALL_DEACYLATE_SWAP")
                                    )

def apply_rulebook(
    data_raw,
    node_idx,
    families=None):
    families = families or [
        "SP2_POLAR_ALL",
        "SP3_POLAR_ALL",
        "SP2_APOLAR_ALL",
        "SP3_APOLAR_ALL",
        "SP2_REACTIVE_ALL",
        "SP3_REACTIVE_ALL",
        "REDOX_FAMILY",
        "ACYL_FAMILY_ALL",
        "AMIDE_FAMILY_ALL",
        "CARBAMATE_FAMILY_ALL",
        "SULFURE_FAMILY_ALL",
        "PHOSPHORUS_FAMILY_ALL",
        "TOGGLE_CHARGE_FAMILY_ALL",
        "POLYVALENT_FAMILY_ALL",
        "ALIPHATIC_FAMILY_ALL",
        "TOGGLE_RING_FAMILY_ALL",
        "RING_FAMILY_ALL",
        "BOND_FAMILY_ALL",
        "DIARYL_FAMILY_ALL"
        ]
    out = []
    if ("SP2_POLAR_ALL" in families or
        "SP3_POLAR_ALL" in families or
        "SP2_APOLAR_ALL" in families or
        "SP3_APOLAR_ALL" in families or
        "SP2_REACTIVE_ALL" in families or
        "SP3_REACTIVE_ALL" in families or
        "REDOX_FAMILY" in families or
        "ACYL_FAMILY_ALL" in families or
        "AMIDE_FAMILY_ALL" in families or
        "CARBAMATE_FAMILY_ALL" in families or
        "SULFURE_FAMILY_ALL" in families or
        "TOGGLE_CHARGE_FAMILY_ALL" in families or
        "POLYVALENT_FAMILY_ALL" in families or
        "ALIPHATIC_FAMILY_ALL" in families or
        "TOGGLE_RING_FAMILY_ALL" in families or
        "RING_FAMILY_ALL" in families or
        "BOND_FAMILY_ALL" in families or
        "DIARYL_FAMILY_ALL" in families):

        seen = set()
        adj = adjacency_list(
            data_raw.edge_index, data_raw.x.size(0))
        attachment_sp2 = is_sp2_attachment(data_raw, node_idx)
        attachment_sp3 = is_sp3_attachment(data_raw, node_idx)
        attachment_adj_carb = adjacent_to_carbonyl(data_raw, node_idx)
        attachment_amidine = is_amidine_core(data_raw, node_idx)
        atoms = ATOMIC_NUMBER()
        arom_idx = len(atoms) + 8 + len(
            CHARGE_CATEGORIES) + len(HYBRIDIZATION_CATEGORIES)
        attachment_aromatic = bool(data_raw.x[node_idx, arom_idx].item() > 0.5)
        reactive_covers_node = False
        cand_attach = {node_idx}
        for n1 in adj[node_idx]:
            cand_attach.add(n1)
            for n2 in adj[n1]:
                cand_attach.add(n2)
        reactive_rules = sp2_reactive_rules() + sp3_reactive_rules()
        for attach in cand_attach:
            for nbr in adj[attach]:
                for rule in reactive_rules:
                    left, _ = rule_fragments(rule["direction"])
                    if not left:
                        continue
                    if not match_fragment(data_raw, attach, nbr, left):
                        continue
                    branch = branch_nodes(data_raw, attach, nbr)
                    if node_idx in branch:
                        reactive_covers_node = True
                        break
                if reactive_covers_node:
                    break
            if reactive_covers_node:
                break
        neighbor_flags = {}
        for nbr in adj[node_idx]:
            neighbor_flags[nbr] = {
                "adjacent_carbonyl": adjacent_to_carbonyl(data_raw, nbr),
                "acyl_link": is_acyl_hetero_link(data_raw, node_idx, nbr),
                "is_cf3_f": is_cf3_f(data_raw, nbr),
                "is_amidine": is_amidine_core(data_raw, nbr)
                }
        def apply_family(
            rules,
            attachment_ok,
            require_aromatic=False,
            block_adjacent_carbonyl=True,
            exclude_nodes=None,
            block_if_sp2_apolar=False,
            block_if_sp2_polar=False,
            block_if_reactive=False):

            if not attachment_ok:
                return
            if require_aromatic and not attachment_aromatic:
                return
            exclude_nodes = set(exclude_nodes or [])
            if block_if_reactive and reactive_covers_node:
                return
            if block_if_sp2_apolar and attachment_sp3:
                for nbr in adj[node_idx]:
                    if not is_sp2_attachment(data_raw, nbr):
                        continue
                    nbr_aromatic = bool(data_raw.x[nbr, arom_idx].item() > 0.5)

                    for rule in sp2_apolar_rules():
                        left, right = rule_fragments(rule["direction"])
                        if not left or not right:
                            continue
                        if not nbr_aromatic:
                            continue
                        if not match_fragment(data_raw, nbr, node_idx, left):
                            continue
                        data_work = copy.copy(data_raw)
                        data_work.x = data_raw.x.clone()
                        data_work.edge_index = data_raw.edge_index.clone()
                        data_work.edge_attr = data_raw.edge_attr.clone()
                        new_data, _ = replace_fragment(
                            data_work, nbr, node_idx, right, copy_data=False
                            )
                        if new_data is not None:
                            return
            if block_if_sp2_polar and attachment_sp3:
                for nbr in adj[node_idx]:
                    if not is_sp2_attachment(data_raw, nbr):
                        continue
                    nbr_aromatic = bool(
                        data_raw.x[nbr, arom_idx].item() > 0.5)
                    for rule in sp2_polar_rules():
                        left, right = rule_fragments(rule["direction"])
                        if not left or not right:
                            continue
                        if not nbr_aromatic:
                            continue
                        if not sp2_polar_filters(
                            data_raw, nbr, node_idx, left, right):
                            continue
                        if not match_fragment(
                            data_raw, nbr, node_idx, left):
                            continue
                        data_work = copy.copy(data_raw)
                        data_work.x = data_raw.x.clone()
                        data_work.edge_index = data_raw.edge_index.clone()
                        data_work.edge_attr = data_raw.edge_attr.clone()
                        new_data, _ = replace_fragment(
                            data_work, nbr, node_idx, right, copy_data=False
                            )
                        if new_data is not None:
                            return
            for rule in rules:
                left, right = rule_fragments(rule["direction"])
                if not left or not right:
                    continue
                if (left == "COOH" and right == "COOMe") or (
                    left == "COOMe" and right == "COOH"):
                    continue
                for nbr in adj[node_idx]:
                    flags = neighbor_flags[nbr]
                    if block_adjacent_carbonyl:
                        if attachment_adj_carb or flags["adjacent_carbonyl"]:
                            continue
                    if left in ("OH", "OMe", "SH") and flags["acyl_link"]:
                        continue
                    if left == "F" and flags["is_cf3_f"]:
                        continue
                    if right in ("F", "Cl", "Br", "I"):
                        if attachment_amidine or flags["is_amidine"]:
                            continue
                    if not match_fragment(data_raw, node_idx, nbr, left):
                        continue
                    if nbr in exclude_nodes:
                        continue
                    data_work = copy.copy(data_raw)
                    data_work.x = data_raw.x.clone()
                    data_work.edge_index = data_raw.edge_index.clone()
                    data_work.edge_attr = data_raw.edge_attr.clone()
                    new_data, removed = replace_fragment(
                        data_work, node_idx, nbr, 
                        right, copy_data=False
                        )
                    if new_data is None:
                        continue
                    if right == "NO2":
                        adj_local = adjacency_list(
                            new_data.edge_index, new_data.x.size(0)
                            )
                        frag = FRAGMENTS.get("NO2")
                        if frag:
                            frag_len = len(frag.get("atoms", []))
                        else:
                            frag_len = 0
                        n_new = new_data.x.size(0)
                        new_nodes = list(range(max(n_new - frag_len, 0), n_new))
                        n_nodes = [n for n in new_nodes if atomic_num_at(new_data, n) == 7]
                        if n_nodes:
                            n_idx = n_nodes[0]
                            o_nodes = [
                                n for n in adj_local[n_idx]
                                if atomic_num_at(new_data, n) == 8
                                ]
                            if len(o_nodes) >= 2:
                                set_formal_charge(new_data, n_idx, 1)
                                set_formal_charge(new_data, o_nodes[0], -1)
                                set_formal_charge(new_data, o_nodes[1], -1)
                    if right in ("sulfonate", "phosphonate_acid"):
                        frag = FRAGMENTS.get(right)
                        if frag:
                            frag_len = len(frag.get("atoms", []))
                            n_new = new_data.x.size(0)
                            new_nodes = list(range(max(n_new - frag_len, 0), n_new))
                            adj_local = adjacency_list(
                                new_data.edge_index, new_data.x.size(0)
                                )
                            for n_idx in new_nodes:
                                if atomic_num_at(new_data, n_idx) != 8:
                                    continue
                                if any(
                                    bond_order_between(new_data, n_idx, nbr) == 2.0
                                    for nbr in adj_local[n_idx]):
                                    continue
                                set_formal_charge(new_data, n_idx, -1)
                    h = (new_data.x.cpu().numpy().tobytes(),
                        new_data.edge_index.cpu().numpy().tobytes(),
                        new_data.edge_attr.cpu().numpy().tobytes()
                        )
                    if h in seen:
                        continue
                    seen.add(h)
                    out.append((new_data, removed, rule["rule_id"]))

        def set_bond_order(data, i, j, order):
            src, dst = data.edge_index
            mask = ((src == i) & (dst == j)) | ((src == j) & (dst == i))
            idxs = torch.where(mask)[0]
            if idxs.numel() == 0:
                return False
            e_idx = int(idxs[0].item())
            data.edge_attr[e_idx, :4] = 0.0
            if order == 1.0:
                data.edge_attr[e_idx, 0] = 1.0
            elif order == 2.0:
                data.edge_attr[e_idx, 1] = 1.0
            elif order == 3.0:
                data.edge_attr[e_idx, 2] = 1.0
            else:
                data.edge_attr[e_idx, 3] = 1.0
            normalize_edge_attr(data)
            return True

        apply_polar_families(
            families,
            attachment_sp2,
            attachment_sp3,
            apply_family
            )
        apply_apolar_families(
            families,
            attachment_sp2,
            attachment_sp3,
            attachment_aromatic,
            node_idx,
            adj,
            data_raw,
            apply_family
            )
        apply_reactive_families(
            families,
            attachment_sp2,
            attachment_sp3,
            apply_family
            )
        apply_redox_family(
            families,
            data_raw,
            node_idx,
            adj,
            out,
            seen
            )
        apply_acyl_family(
            families,
            data_raw,
            node_idx,
            adj,
            attachment_sp2,
            attachment_sp3,
            out,
            seen
            )
        apply_amide_family(
            families,
            data_raw,
            node_idx,
            attachment_sp2,
            attachment_sp3,
            out,
            seen
            )
        apply_carbamate_family(
            families,
            data_raw,
            node_idx,
            attachment_sp2,
            attachment_sp3,
            out,
            seen
            )
        apply_sulfure_family(
            families,
            data_raw,
            node_idx,
            out,
            seen
            )
        apply_phosphorus_family(
            families,
            data_raw,
            node_idx,
            out,
            seen
            )
        apply_toggle_ring_family(
            families,
            data_raw,
            node_idx,
            out,
            seen
            )
        apply_ring_family(
            families,
            data_raw,
            node_idx,
            out,
            seen
            )
        apply_bond_family(
            families,
            data_raw,
            node_idx,
            out,
            seen,
            set_bond_order
            )
        apply_diaryl_family(
            families,
            data_raw,
            node_idx,
            out,
            seen
            )
        apply_aliphatic_family(
            families,
            data_raw,
            node_idx,
            out,
            seen
            )
    return out
