import torch
from rdkit import Chem

from graphs import ATOMIC_NUMBER


CHARGE_CATEGORIES = [-1, 0, 1, 2, 3, 4]
DEGREE_CATEGORIES = list(range(8))
HYBRIDIZATION_CATEGORIES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
    ]


MAX_VALENCE = {
    1: 1, 5: 3, 6: 4, 7: 3, 8: 2, 9: 1,
    14: 4, 15: 5, 16: 6, 17: 1, 33: 5,
    34: 6, 35: 1, 53: 1
    }


RING_MAX_VALENCE = {
    6: 3,
    7: 3,
    8: 2,
    16: 2,
    34: 2
    }


SP2_POLAR_ATOMS = [7, 8, 9, 16]
SP2_POLAR_CHARGE = {
    7: [-1, 0, 1],
    8: [-1, 0],
    9: [-1, 0],
    16: [-1, 0, 1]
    }


POLAR_FRAGMENTS = [
    "F", "OH", "SH", "NH2", "NHMe", 
    "NMe2", "CF2", "OCF2", "NO2"
    ]

APOLAR_FRAGMENTS = [
    "Me", "Cl", "Br", "I", "OMe", "Et", "iPr", 
    "tBu", "CF3", "OCF3", "CCl3", "SMe"
    ]

REACTIVE_FRAGMENTS = [
    "CN", "CHO",
    "NO2", "NCO", "N3", "N2",
    "vinylF", "vinylCl", "vinylBr", "vinylI"
    ]

ACYL_FAMILY_FRAGMENTS = [
    "COOH", "COOMe",
    "ketone", "thioketone", "selenoketone", "thioester"
    ]

AMIDE_FAMILY_FRAGMENTS = [
    "CONH2", "thioamide", "selenoamide",
    "urea", "thiourea", "selenourea", 
    "guanidine", "amidine"
    ]

CARBAMATE_FAMILY_FRAGMENTS = [
    "COOMe", "carbonate", "carbamate",
    "thiocarbamate", "selenocarbamate", "thioester"
    ]

SULFONE_SULFONAMIDE_FRAGMENTS = [
    "sulfone", "sulfonamide", "sulfonate",
    "sulfoxide", "sulfilimine", 
    "sulfoximine", "sulfondiimine",
    "sulfonimidamide", "sulfondiimidamide",
    "boronate"
    ]


FRAGMENTS = {
    "Me": {"atoms": [6], 
        "edges": [], 
        "attach_idx": 0
    },
    "F": {"atoms": [9], 
        "edges": [], 
        "attach_idx": 0
    },
    "Cl": {"atoms": [17], 
        "edges": [], 
        "attach_idx": 0
    },
    "Br": {"atoms": [35], 
        "edges": [], 
        "attach_idx": 0
    },
    "I": {"atoms": [53], 
        "edges": [], 
        "attach_idx": 0
    },
    "OH": {"atoms": [8], 
        "edges": [], 
        "attach_idx": 0
    },
    "SH": {"atoms": [16], 
        "edges": [], 
        "attach_idx": 0
    },
    "SMe": {"atoms": [16, 6], 
        "edges": [(0, 1, 1.0)], 
        "attach_idx": 0
    },
    "OMe": {"atoms": [8, 6], 
        "edges": [(0, 1, 1.0)], 
        "attach_idx": 0
    },
    "Et": {"atoms": [6, 6], 
        "edges": [(0, 1, 1.0)], 
        "attach_idx": 0
    },
    "iPr": {"atoms": [6, 6, 6], 
        "edges": [(0, 1, 1.0), (0, 2, 1.0)], 
        "attach_idx": 0
    },
    "tBu": {"atoms": [6, 6, 6, 6], 
        "edges": [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)], 
        "attach_idx": 0
    },
    "CF3": {"atoms": [6, 9, 9, 9], 
        "edges": [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)], 
        "attach_idx": 0
    },
    "CF2": {"atoms": [6, 9, 9], 
        "edges": [(0, 1, 1.0), (0, 2, 1.0)], 
        "attach_idx": 0
    },
    "OCF2": {"atoms": [8, 6, 9, 9], 
        "edges": [(0, 1, 1.0), (1, 2, 1.0), (1, 3, 1.0)], 
        "attach_idx": 0
    },
    "OCF3": {"atoms": [8, 6, 9, 9, 9], 
        "edges": [(0, 1, 1.0), (1, 2, 1.0), (1, 3, 1.0), (1, 4, 1.0)], 
        "attach_idx": 0
    },
    "CCl3": {"atoms": [6, 17, 17, 17], 
        "edges": [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)], 
        "attach_idx": 0
    },
    "NH2": {"atoms": [7], 
        "edges": [], 
        "attach_idx": 0
    },
    "NHMe": {"atoms": [7, 6], 
        "edges": [(0, 1, 1.0)], 
        "attach_idx": 0
    },
    "NMe2": {"atoms": [7, 6, 6], 
        "edges": [(0, 1, 1.0), (0, 2, 1.0)], 
        "attach_idx": 0
    },
    "CN": {"atoms": [6, 7], 
        "edges": [(0, 1, 3.0)], 
        "attach_idx": 0
    },
    "CHO": {"atoms": [6, 8], 
        "edges": [(0, 1, 2.0)], 
        "attach_idx": 0
    },
    "NO2": {"atoms": [7, 8, 8], 
        "edges": [(0, 1, 2.0), (0, 2, 1.0)], 
        "attach_idx": 0
    },
    "NCO": {"atoms": [7, 6, 8], 
        "edges": [(0, 1, 2.0), (1, 2, 2.0)], 
        "attach_idx": 0
    },
    "N3": {"atoms": [7, 7, 7], 
        "edges": [(0, 1, 2.0), (1, 2, 2.0)], 
        "attach_idx": 0
    },
    "N2": {"atoms": [7, 7], 
        "edges": [(0, 1, 3.0)], 
        "attach_idx": 0
    },
    "vinylF": {"atoms": [6, 6, 9], 
        "edges": [(0, 1, 2.0), (1, 2, 1.0)], 
        "attach_idx": 0
    },
    "vinylCl": {"atoms": [6, 6, 17], 
        "edges": [(0, 1, 2.0), (1, 2, 1.0)], 
        "attach_idx": 0
    },
    "vinylBr": {
        "atoms": [6, 6, 35], 
        "edges": [(0, 1, 2.0), (1, 2, 1.0)], 
        "attach_idx": 0
    },
    "vinylI": {
        "atoms": [6, 6, 53], 
        "edges": [(0, 1, 2.0), (1, 2, 1.0)], 
        "attach_idx": 0
    },
    "COOH": {
        "atoms": [6, 8, 8], 
        "edges": [(0, 1, 2.0), (0, 2, 1.0)], 
        "attach_idx": 0
    },
    "COOMe": {"atoms": [6, 8, 8, 6], 
              "edges": [(0, 1, 2.0), (0, 2, 1.0), (2, 3, 1.0)], 
              "attach_idx": 0
    },
    "COOMe_inv": {
        "atoms": [8, 6, 8, 8, 6], 
        "edges": [(0, 1, 1.0), (1, 2, 2.0), (1, 3, 1.0), (3, 4, 1.0)], 
        "attach_idx": 0
    },
    "CONH2": {"atoms": [6, 8, 7], 
        "edges": [(0, 1, 2.0), (0, 2, 1.0)], 
        "attach_idx": 0
    },
    "CONH2_inv": {
        "atoms": [7, 6, 8, 6], 
        "edges": [(0, 1, 1.0), (1, 2, 2.0), (1, 3, 1.0)], 
        "attach_idx": 0
    },
    "ketone": {
        "atoms": [6, 8], 
        "edges": [(0, 1, 2.0)], 
        "attach_idx": 0
    },
    "ketone_sub": {
        "atoms": [6, 8, 6], 
        "edges": [(0, 1, 2.0), (0, 2, 1.0)], 
        "attach_idx": 0
    },
    "ketone_inv": {
        "atoms": [6, 6, 8, 6], 
        "edges": [(0, 1, 1.0), (1, 2, 2.0), (1, 3, 1.0)], 
        "attach_idx": 0
    },
    "thioketone": {
        "atoms": [6, 16], 
        "edges": [(0, 1, 2.0)], 
        "attach_idx": 0
    },
    "thioketone_sub": {
        "atoms": [6, 16, 6], 
        "edges": [(0, 1, 2.0), (0, 2, 1.0)], 
        "attach_idx": 0
    },
    "thioketone_inv": {
        "atoms": [6, 6, 16, 6], 
        "edges": [(0, 1, 1.0), (1, 2, 2.0), (1, 3, 1.0)], 
        "attach_idx": 0
    },
    "selenoketone": {
        "atoms": [6, 34], 
        "edges": [(0, 1, 2.0)], 
        "attach_idx": 0
    },
    "selenoketone_sub": {
        "atoms": [6, 34, 6], 
        "edges": [(0, 1, 2.0), (0, 2, 1.0)], 
        "attach_idx": 0
    },
    "selenoketone_inv": {
        "atoms": [6, 6, 34, 6], 
        "edges": [(0, 1, 1.0), (1, 2, 2.0), (1, 3, 1.0)],
        "attach_idx": 0
    },
    "thioester": {
        "atoms": [6, 8, 16], 
        "edges": [(0, 1, 2.0), (0, 2, 1.0)], 
        "attach_idx": 0
    },
    "thioester_inv": {
        "atoms": [16, 6, 8, 6], 
        "edges": [(0, 1, 1.0), (1, 2, 2.0), (1, 3, 1.0)], 
        "attach_idx": 0
    },
    "sulfone": {
        "atoms": [16, 8, 8, 6],
        "edges": [(0, 1, 2.0), (0, 2, 2.0), (0, 3, 1.0)],
        "attach_idx": 0
    },
    "sulfonamide": {
        "atoms": [16, 8, 8, 7, 6],
        "edges": [(0, 1, 2.0), (0, 2, 2.0), (0, 3, 1.0), (3, 4, 1.0)],
        "attach_idx": 0
    },
    "sulfonate": {
        "atoms": [16, 8, 8, 8],
        "edges": [(0, 1, 2.0), (0, 2, 2.0), (0, 3, 1.0)],
        "attach_idx": 0
    },
    "sulfonylurea": {
        "atoms": [16, 8, 8, 7, 6, 8, 7, 6],
        "edges": [
            (0, 1, 2.0), (0, 2, 2.0), (0, 3, 1.0),
            (3, 4, 1.0), (4, 5, 2.0), (4, 6, 1.0),
            (6, 7, 1.0)],
        "attach_idx": 0
    },
    "phosphonate": {
        "atoms": [15, 8, 8, 6, 8, 6],
        "edges": [(0, 1, 2.0), (0, 2, 1.0), 
                  (2, 3, 1.0), (0, 4, 1.0), (4, 5, 1.0)],
        "attach_idx": 0
    },
    "phosphonate_acid": {
        "atoms": [15, 8, 8, 8],
        "edges": [(0, 1, 2.0), (0, 2, 1.0), (0, 3, 1.0)],
        "attach_idx": 0
    },
    "phosphona": {
        "atoms": [15, 8, 6, 6],
        "edges": [(0, 1, 2.0), (0, 2, 1.0), (0, 3, 1.0)],
        "attach_idx": 0
    },
    "boronate": {
        "atoms": [5, 8, 6, 8, 6],
        "edges": [(0, 1, 1.0), (1, 2, 1.0), (0, 3, 1.0), (3, 4, 1.0)],
        "attach_idx": 0}
    }


def feature_slices():
    atoms = ATOMIC_NUMBER()
    n_atoms = len(atoms)
    d_start = n_atoms
    c_start = d_start + len(DEGREE_CATEGORIES)
    h_start = c_start + len(CHARGE_CATEGORIES)
    a_start = h_start + len(HYBRIDIZATION_CATEGORIES)
    
    return {
        "atoms": atoms,
        "atomic": (0, n_atoms),
        "degree": (d_start, c_start),
        "charge": (c_start, h_start),
        "hybrid": (h_start, a_start),
        "aromatic": a_start
        }


def bond_orders(edge_attr):
    bond_type = edge_attr[:, :4]
    bond_idx = bond_type.argmax(dim=1)
    order_map = bond_idx.new_tensor(
        [1.0, 2.0, 3.0, 1.5], dtype=edge_attr.dtype)
    
    return order_map[bond_idx]


def adjacency_list(edge_index, n_nodes):
    adj = [[] for _ in range(n_nodes)]
    src, dst = edge_index
    for i, j in zip(src.tolist(), dst.tolist()):
        adj[i].append(j)
        adj[j].append(i)

    return adj


def node_valence_degree(data, node_idx):
    if data.edge_attr is None or data.edge_attr.numel() == 0:
        return 0.0, 0
    src, dst = data.edge_index
    orders = bond_orders(data.edge_attr)
    mask = (src == node_idx) | (dst == node_idx)
    if not mask.any():
        return 0.0, 0
    valence = float(orders[mask].sum().item())
    degree = int(mask.sum().item())

    return valence, degree


def bond_order_between(data, i, j):
    src, dst = data.edge_index
    mask = ((src == i) & (dst == j)) | ((src == j) & (dst == i))
    idxs = torch.where(mask)[0]
    if idxs.numel() == 0:
        return None
    orders = bond_orders(data.edge_attr)

    return float(orders[int(idxs[0].item())].item())


def atomic_num_at(data, node_idx):
    slices = feature_slices()
    atoms = slices["atoms"]
    a_start, a_end = slices["atomic"]

    return atoms[int(data.x[
        node_idx, a_start:a_end].argmax().item())
        ]


def atomic_charge_at(data, node_idx):
    slices = feature_slices()
    c_start, c_end = slices["charge"]
    charge_idx = int(data.x[
        node_idx, c_start:c_end].argmax().item())

    return CHARGE_CATEGORIES[charge_idx]


def is_ring_atom(data, idx):
    if data.edge_attr is None or data.edge_attr.numel() == 0:
        return False
    if data.edge_attr.size(1) < 6:
        return False
    src, dst = data.edge_index
    mask = (src == idx) | (dst == idx)
    if not mask.any():
        return False
    
    return bool((
        data.edge_attr[mask, 5] > 0.5).any().item()
        )


def branch_nodes(data, attachment, neighbor):
    n = data.x.size(0)
    adj = adjacency_list(data.edge_index, n)
    seen = set([attachment])
    stack = [neighbor]
    nodes = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        nodes.add(cur)
        for nxt in adj[cur]:
            if nxt not in seen:
                stack.append(nxt)

    return nodes


def is_sp2_attachment(data, node_idx):
    valence, _ = node_valence_degree(data, node_idx)
    slices = feature_slices()
    arom_idx = slices["aromatic"]
    aromatic = bool(data.x[node_idx, arom_idx].item() > 0.5)
    
    return aromatic or valence >= 3.0


def is_sp3_attachment(data, node_idx):
    atom_num = atomic_num_at(data, node_idx)
    if atom_num != 6:
        return False
    slices = feature_slices()
    arom_idx = slices["aromatic"]
    if bool(data.x[node_idx, arom_idx].item() > 0.5):
        return False
    valence, _ = node_valence_degree(data, node_idx)

    return valence <= 4.0


def is_carbonyl_c(data, node_idx):
    if data.edge_attr is None or data.edge_attr.numel() == 0:
        return False
    slices = feature_slices()
    atoms = slices["atoms"]
    a_start, a_end = slices["atomic"]
    atom_num = atoms[int(data.x[
        node_idx, a_start:a_end].argmax().item())]
    if atom_num != 6:
        return False
    src, dst = data.edge_index
    orders = bond_orders(data.edge_attr)
    mask = (src == node_idx) | (dst == node_idx)
    for k in torch.where(mask)[0].tolist():
        nbr = int(dst[k].item()) if int(src[k].item()
                ) == node_idx else int(src[k].item())
        nbr_atom = atoms[int(
            data.x[nbr, a_start:a_end].argmax().item())]
        if float(orders[k].item()
                ) >= 2.0 and nbr_atom in (8, 16, 34, 7):
            return True
        
    return False


def adjacent_to_carbonyl(data, node_idx):
    if is_carbonyl_c(data, node_idx):
        return True
    adj = adjacency_list(data.edge_index, data.x.size(0))
    for nbr in adj[node_idx]:
        if is_carbonyl_c(data, nbr):
            return True
        
    return False


def is_acyl_hetero_link(data, attachment, neighbor):
    def is_carbonyl(idx):
        return is_carbonyl_c(data, idx)

    slices = feature_slices()
    atoms = slices["atoms"]
    a_start, a_end = slices["atomic"]
    a_atom = atoms[int(data.x[attachment, 
                a_start:a_end].argmax().item())]
    n_atom = atoms[int(data.x[neighbor, 
                a_start:a_end].argmax().item())]
    if is_carbonyl(attachment) and n_atom in (7, 8, 16):
        return True
    if is_carbonyl(neighbor) and a_atom in (7, 8, 16):
        return True
    
    return False


def is_cf3_f(data, node_idx):
    slices = feature_slices()
    atoms = slices["atoms"]
    a_start, a_end = slices["atomic"]
    atom_num = atoms[int(data.x[
        node_idx, a_start:a_end].argmax().item())]
    if atom_num != 9:
        return False
    _, degree = node_valence_degree(data, node_idx)
    if degree != 1:
        return False
    adj = adjacency_list(data.edge_index, data.x.size(0))
    if not adj[node_idx]:
        return False
    c = adj[node_idx][0]
    c_atom = atoms[int(data.x[
        c, a_start:a_end].argmax().item())]
    if c_atom != 6:
        return False
    f_neighbors = 0
    for nbr in adj[c]:
        nbr_atom = atoms[int(data.x[
            nbr, a_start:a_end].argmax().item())]
        if nbr_atom == 9:
            if bond_order_between(data, c, nbr) != 1.0:
                return False
            f_neighbors += 1

    return f_neighbors == 3


def is_polyhalogen_on_carbon(data, node_idx):
    atom_num = atomic_num_at(data, node_idx)
    if atom_num not in (9, 17, 35, 53):
        return False
    _, degree = node_valence_degree(data, node_idx)
    if degree != 1:
        return False
    adj = adjacency_list(data.edge_index, data.x.size(0))
    if not adj[node_idx]:
        return False
    c = adj[node_idx][0]
    if atomic_num_at(data, c) != 6:
        return False
    if bond_order_between(data, c, node_idx) != 1.0:
        return False
    halogen_neighbors = 0
    for nbr in adj[c]:
        if atomic_num_at(data, nbr) in (9, 17, 35, 53):
            if bond_order_between(data, c, nbr) != 1.0:
                return False
            halogen_neighbors += 1

    return halogen_neighbors >= 2


def branch_has_cycle(data, nodes):
    if not nodes:
        return False
    node_set = set(nodes)
    if len(node_set) <= 2:
        return False
    edge_count = 0
    src, dst = data.edge_index
    for i, j in zip(src.tolist(), dst.tolist()):
        if i in node_set and j in node_set:
            edge_count += 1
    edge_count //= 2

    return edge_count >= len(node_set)


def is_amidine_core(data, node_idx):
    slices = feature_slices()
    atoms = slices["atoms"]
    a_start, a_end = slices["atomic"]
    atom_num = atoms[
        int(data.x[node_idx, a_start:a_end].argmax().item())]
    if atom_num != 6:
        return False
    if data.edge_attr is None or data.edge_attr.numel() == 0:
        return False
    src, dst = data.edge_index
    orders = bond_orders(data.edge_attr)
    mask = (src == node_idx) | (dst == node_idx)
    if not mask.any():
        return False
    double_n = 0
    single_n = 0
    single_c = 0
    for k in torch.where(mask)[0].tolist():
        nbr = int(dst[k].item()) if int(
            src[k].item()) == node_idx else int(src[k].item())
        order = float(orders[k].item())
        nbr_atom = atoms[int(data.x[nbr, a_start:a_end].argmax().item())]
        if order == 2.0 and nbr_atom == 7:
            double_n += 1
        elif order == 1.0 and nbr_atom == 7:
            single_n += 1
        elif order == 1.0 and nbr_atom == 6:
            single_c += 1

    return double_n == 1 and single_n == 1 and single_c >= 1


def match_fragment(data, attachment, neighbor, frag_key):
    frag = FRAGMENTS.get(frag_key)
    if frag is None:
        return False
    if frag_key == "OH":
        if atomic_num_at(data, neighbor) != 8:
            return False
        _, degree = node_valence_degree(data, neighbor)
        if degree != 1:
            return False
        if bond_order_between(
            data, attachment, neighbor) != 1.0:
            return False
        return True
    if frag_key == "OMe":
        if atomic_num_at(data, neighbor) != 8:
            return False
        _, degree = node_valence_degree(data, neighbor)
        if degree != 2:
            return False
        adj = adjacency_list(
            data.edge_index, data.x.size(0))
        other = [n for n in adj[neighbor] if n != attachment]
        if len(other) != 1:
            return False
        other_idx = other[0]
        if atomic_num_at(
            data, other_idx) != 6:
            return False
        _, degree_c = node_valence_degree(data, other_idx)
        if degree_c != 1:
            return False
        if bond_order_between(
            data, neighbor, other_idx) != 1.0:
            return False
        return True
    if frag_key == "OCF3":
        if atomic_num_at(data, neighbor) != 8:
            return False
        _, degree = node_valence_degree(data, neighbor)
        if degree != 2:
            return False
        adj = adjacency_list(
            data.edge_index, data.x.size(0))
        other = [n for n in adj[neighbor] if n != attachment]
        if len(other) != 1:
            return False
        c_idx = other[0]
        if atomic_num_at(data, c_idx) != 6:
            return False
        if bond_order_between(
            data, neighbor, c_idx) != 1.0:
            return False
        adj_c = [n for n in adj[c_idx] if n != neighbor]
        if len(adj_c) != 3:
            return False
        for f_idx in adj_c:
            if atomic_num_at(
                data, f_idx) != 9:
                return False
            if bond_order_between(data, c_idx, f_idx) != 1.0:
                return False
        return True
    if frag_key in ("ketone", "thioketone", "selenoketone"):
        if atomic_num_at(data, attachment) != 6:
            return False
        if atomic_num_at(data, neighbor) != 6:
            return False
        if bond_order_between(data, attachment, neighbor) != 1.0:
            return False
        adj = adjacency_list(
            data.edge_index, data.x.size(0))
        hetero = {"ketone": 8, "thioketone": 16, 
                  "selenoketone": 34}[frag_key]
        has_double = False
        has_other_c = False
        for nbr in adj[neighbor]:
            order = bond_order_between(data, neighbor, nbr)
            if nbr == attachment:
                continue
            if order == 2.0 and atomic_num_at(data, nbr) == hetero:
                has_double = True
            if order == 1.0:
                nbr_atom = atomic_num_at(data, nbr)
                if nbr_atom == 6:
                    has_other_c = True
                if nbr_atom in (7, 8, 16, 34):
                    return False
        if not has_double:
            return False
        if not has_other_c:
            return False
        return True
    if frag_key == "CHO":
        if atomic_num_at(
            data, attachment) != 6:
            return False
        if atomic_num_at(
            data, neighbor) != 6:
            return False
        if bond_order_between(
            data, attachment, neighbor) != 1.0:
            return False
        adj = adjacency_list(
            data.edge_index, data.x.size(0))
        has_double_o = False
        has_other_c = False
        for nbr in adj[neighbor]:
            order = bond_order_between(
                data, neighbor, nbr)
            if nbr == attachment:
                continue
            if order == 2.0 and atomic_num_at(data, nbr) == 8:
                has_double_o = True
            if order == 1.0 and atomic_num_at(data, nbr) == 6:
                has_other_c = True
            if order == 1.0 and atomic_num_at(data, nbr) in (7, 8, 16, 34):
                return False
        if not has_double_o:
            return False
        if has_other_c:
            return False
        return True
    if frag_key in (
        "sulfone", "sulfonamide", "sulfonate",
        "sulfoxide", "sulfilimine", 
        "sulfoximine", "sulfondiimine",
        "sulfonimidamide", "sulfondiimidamide"):

        return match_from_sulfone(
            data, attachment, neighbor, frag_key
            )
    if frag_key in ("F", "Cl", "Br", "I"):
        if is_polyhalogen_on_carbon(data, neighbor):
            return False
    nodes = branch_nodes(data, attachment, neighbor)

    if frag_key in ("Me", "Et", "iPr", "tBu", "CF3", 
        "CF2", "CCl3", "SMe", "OMe", "OCF2", "OCF3"):

        if is_ring_atom(data, attachment
            ) and is_ring_atom(data, neighbor):
            return False
        if branch_has_cycle(data, nodes):
            return False
    if len(nodes) != len(frag["atoms"]):
        return False
    node_list = list(nodes)
    atom_nums = [atomic_num_at(data, n) for n in node_list]
    if sorted(atom_nums) != sorted(frag["atoms"]):
        return False
    if len(frag["atoms"]) == 1:
        if bond_order_between(data, attachment, neighbor) != 1.0:
            return False
        if frag_key == "Me":
            if atomic_num_at(data, attachment) == 6 and not is_ring_atom(data, attachment):
                adj = adjacency_list(data.edge_index, data.x.size(0))
                carbon_neighbors = [
                    n for n in adj[attachment]
                    if n != neighbor and atomic_num_at(data, n) == 6
                    and bond_order_between(data, attachment, n) == 1.0
                    ]
                if len(carbon_neighbors) > 1:
                    return False
        return True
    frag_atoms = frag["atoms"]
    frag_edges = {(i, j): order for i, 
            j, order in frag["edges"]}
    frag_edges.update({(j, i): order for i, 
            j, order in frag["edges"]})
    candidates = {}
    for i, a in enumerate(frag_atoms):
        candidates[i] = [n for n in node_list if atomic_num_at(data, n) == a]
    attach_idx = frag.get("attach_idx")
    if attach_idx is not None:
        candidates[attach_idx] = [
            n for n in candidates.get(attach_idx, []) if n == neighbor
        ]
        if not candidates[attach_idx]:
            return False

    used = set()
    assign = {}

    def is_compatible(i, n):
        for j, m in assign.items():
            order = frag_edges.get((i, j))
            bond = bond_order_between(data, n, m)
            if order is None and bond is not None:
                return False
            if order is not None and bond != order:
                return False
        return True

    def backtrack(idx):
        if idx == len(frag_atoms):
            for i in range(len(frag_atoms)):
                for j in range(i + 1, len(frag_atoms)):
                    n = assign[i]
                    m = assign[j]
                    bond = bond_order_between(data, n, m)
                    if bond is not None and (i, j) not in frag_edges:
                        return False
            return True
        if idx not in candidates:
            return False
        for n in candidates[idx]:
            if n in used:
                continue
            if not is_compatible(idx, n):
                continue
            assign[idx] = n
            used.add(n)
            if backtrack(idx + 1):
                return True
            used.remove(n)
            assign.pop(idx, None)
        return False

    return backtrack(0)


def branch_all_carbons(data, start, blocked):
    adj = adjacency_list(data.edge_index, data.x.size(0))
    seen = set([blocked])
    stack = [start]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if atomic_num_at(data, cur) != 6:
            return False
        for nxt in adj[cur]:
            if nxt not in seen:
                stack.append(nxt)
    return True


def match_from_carbonyl(data, attachment, carbonyl):
    if atomic_num_at(data, carbonyl) != 6:
        return False
    if bond_order_between(data, attachment, carbonyl) != 1.0:
        return False
    adj = adjacency_list(data.edge_index, data.x.size(0))
    has_double_o = None
    single_o = None
    has_single_c = False
    for nbr in adj[carbonyl]:
        if nbr == attachment:
            continue
        order = bond_order_between(data, carbonyl, nbr)
        atom = atomic_num_at(data, nbr)
        if order == 2.0 and atom == 8:
            has_double_o = nbr
        elif order == 1.0 and atom == 8 and single_o is None:
            single_o = nbr
        elif order == 1.0 and atom == 6:
            has_single_c = True
            continue
        else:
            return False
    if has_double_o is None or single_o is None:
        return False
    if not has_single_c:
        return False
    adj_o = [n for n in adj[single_o] if n != carbonyl]
    if len(adj_o) != 1:
        return False
    if atomic_num_at(data, adj_o[0]) != 6:
        return False
    if not branch_all_carbons(data, adj_o[0], single_o):
        return False
    
    return True


def match_from_oxygen(data, attachment, oxy):
    if atomic_num_at(data, oxy) != 8:
        return False
    if bond_order_between(data, attachment, oxy) != 1.0:
        return False
    adj = adjacency_list(data.edge_index, data.x.size(0))
    nbrs = [n for n in adj[oxy] if n != attachment]
    if len(nbrs) != 1:
        return False
    carbonyl = nbrs[0]
    if atomic_num_at(data, carbonyl) != 6:
        return False
    if bond_order_between(data, oxy, carbonyl) != 1.0:
        return False
    has_double_o = None
    acyl_c = None
    for nbr in adj[carbonyl]:
        if nbr == oxy:
            continue
        order = bond_order_between(data, carbonyl, nbr)
        atom = atomic_num_at(data, nbr)
        if order == 2.0 and atom == 8:
            has_double_o = nbr
        elif order == 1.0 and atom == 6 and acyl_c is None:
            acyl_c = nbr
        else:
            return False
    if has_double_o is None or acyl_c is None:
        return False
    if not branch_all_carbons(data, acyl_c, carbonyl):
        return False
    
    return True


def match_carbonyl(data, attachment, carbonyl):
    if atomic_num_at(data, carbonyl) != 6:
        return False
    if bond_order_between(data, attachment, carbonyl) != 1.0:
        return False
    adj = adjacency_list(data.edge_index, data.x.size(0))
    has_double_o = None
    single_s = None
    for nbr in adj[carbonyl]:
        if nbr == attachment:
            continue
        order = bond_order_between(data, carbonyl, nbr)
        atom = atomic_num_at(data, nbr)
        if order == 2.0 and atom == 8:
            has_double_o = nbr
        elif order == 1.0 and atom == 16:
            single_s = nbr
        elif order == 1.0 and atom == 6:
            continue
        else:
            return False
    if has_double_o is None or single_s is None:
        return False
    _, degree_s = node_valence_degree(data, single_s)
    if degree_s > 2:
        return False
    adj_s = [n for n in adj[single_s] if n != carbonyl]
    if len(adj_s) > 1:
        return False
    if adj_s:
        other = adj_s[0]
        if atomic_num_at(data, other) != 6:
            return False
        if bond_order_between(data, single_s, other) != 1.0:
            return False
        
    return True


def match_from_sulfur(data, attachment, sulfur):
    if atomic_num_at(data, sulfur) != 16:
        return False
    if bond_order_between(
        data, attachment, sulfur) != 1.0:
        return False
    _, degree_s = node_valence_degree(data, sulfur)
    if degree_s > 2:
        return False
    adj = adjacency_list(
        data.edge_index, data.x.size(0))
    nbrs = [n for n in adj[sulfur] if n != attachment]
    if len(nbrs) != 1:
        return False
    carbonyl = nbrs[0]
    if atomic_num_at(data, carbonyl) != 6:
        return False
    if bond_order_between(
        data, sulfur, carbonyl) != 1.0:
        return False
    has_double_o = None
    acyl_c = None
    for nbr in adj[carbonyl]:
        if nbr == sulfur:
            continue
        order = bond_order_between(data, carbonyl, nbr)
        atom = atomic_num_at(data, nbr)
        if order == 2.0 and atom == 8:
            has_double_o = nbr
        elif order == 1.0 and atom == 6 and acyl_c is None:
            acyl_c = nbr
        else:
            return False
    if has_double_o is None or acyl_c is None:
        return False
    if not branch_all_carbons(data, acyl_c, carbonyl):
        return False
    
    return True


def match_amide_from_carbonyl(data, attachment, carbonyl):
    if atomic_num_at(data, carbonyl) != 6:
        return False
    if bond_order_between(data, attachment, carbonyl) != 1.0:
        return False
    adj = adjacency_list(data.edge_index, data.x.size(0))
    has_double_o = None
    single_n = None
    for nbr in adj[carbonyl]:
        if nbr == attachment:
            continue
        order = bond_order_between(data, carbonyl, nbr)
        atom = atomic_num_at(data, nbr)
        if order == 2.0 and atom == 8:
            has_double_o = nbr
        elif order == 1.0 and atom == 7:
            single_n = nbr
        elif order == 1.0 and atom == 6:
            continue
        else:
            return False
    if has_double_o is None or single_n is None:
        return False
    _, degree_n = node_valence_degree(data, single_n)
    if degree_n > 3:
        return False
    adj_n = [n for n in adj[single_n] if n != carbonyl]
    for other in adj_n:
        if atomic_num_at(data, other) != 6:
            return False
        if bond_order_between(
            data, single_n, other) != 1.0:
            return False
        
    return True


def match_from_nitrogen(data, attachment, nitrogen):
    if atomic_num_at(data, nitrogen) != 7:
        return False
    if bond_order_between(
        data, attachment, nitrogen) != 1.0:
        return False
    _, degree_n = node_valence_degree(data, nitrogen)
    if degree_n > 3:
        return False
    adj = adjacency_list(
        data.edge_index, data.x.size(0))
    nbrs = [n for n in adj[nitrogen] if n != attachment]
    carbonyl = None
    for n in nbrs:
        if atomic_num_at(data, n
            ) == 6 and bond_order_between(
                data, nitrogen, n) == 1.0:
            carbonyl = n
            break
    if carbonyl is None:
        return False
    has_double_o = None
    acyl_c = None
    for nbr in adj[carbonyl]:
        if nbr == nitrogen:
            continue
        order = bond_order_between(data, carbonyl, nbr)
        atom = atomic_num_at(data, nbr)
        if order == 2.0 and atom == 8:
            has_double_o = nbr
        elif order == 1.0 and atom == 6 and acyl_c is None:
            acyl_c = nbr
        else:
            return False
    if has_double_o is None or acyl_c is None:
        return False
    if not branch_all_carbons(data, acyl_c, carbonyl):
        return False
    
    return True


def match_from_sulfone(data, attachment, sulfur, frag_key):
    if atomic_num_at(data, sulfur) != 16:
        return False
    if bond_order_between(data, attachment, sulfur) != 1.0:
        return False
    adj = adjacency_list(data.edge_index, data.x.size(0))
    double_o = 0
    double_n = 0
    single_c = 0
    single_n = 0
    single_o = 0
    for nbr in adj[sulfur]:
        if nbr == attachment:
            continue
        order = bond_order_between(data, sulfur, nbr)
        atom = atomic_num_at(data, nbr)
        if order == 2.0 and atom == 8:
            double_o += 1
        elif order == 2.0 and atom == 7:
            double_n += 1
        elif order == 1.0 and atom == 6:
            single_c += 1
        elif order == 1.0 and atom == 7:
            single_n += 1
        elif order == 1.0 and atom == 8:
            single_o += 1
        else:
            return False
    double_total = double_o + double_n

    if double_total not in (1, 2):
        return False
    
    if single_c > 2:
        return False
    
    if frag_key == "sulfone":
        return (
            double_o == 2 and
            double_n == 0 and
            single_n == 0 and
            single_o == 0 and
            single_c == 2
            )
    if frag_key == "sulfonamide":
        return (
            double_o == 2 and
            double_n == 0 and
            single_n == 1 and
            single_o == 0 and
            single_c == 1
            )
    if frag_key == "sulfonate":
        return (
            double_o == 2 and
            double_n == 0 and
            single_o == 1 and
            single_n == 0 and
            single_c == 1
            )
    if frag_key == "sulfoxide":
        return (
            double_o == 1 and
            double_n == 0 and
            single_n == 0 and
            single_o == 0 and
            single_c == 2
            )
    if frag_key == "sulfilimine":
        return (
            double_o == 0 and
            double_n == 1 and
            single_n == 0 and
            single_o == 0 and
            single_c == 2
            )
    if frag_key == "sulfoximine":
        return (
            double_o == 1 and
            double_n == 1 and
            single_n == 0 and
            single_o == 0 and
            single_c == 2
            )
    if frag_key == "sulfondiimine":
        return (
            double_o == 0 and
            double_n == 2 and
            single_n == 0 and
            single_o == 0 and
            single_c == 2
            )
    if frag_key == "sulfonimidamide":
        return (
            double_o == 2 and
            double_n == 1 and
            single_n == 1 and
            single_o == 0 and
            single_c == 1
            )
    if frag_key == "sulfondiimidamide":
        return (
            double_o == 0 and
            double_n == 2 and
            single_n == 1 and
            single_o == 0 and
            single_c == 1
            )

    return False


def _carbonyl_single_heteros(data, carbonyl, exclude=None):
    adj = adjacency_list(data.edge_index, data.x.size(0))
    heteros = []
    for nbr in adj[carbonyl]:
        if exclude is not None and nbr == exclude:
            continue
        order = bond_order_between(data, carbonyl, nbr)
        if order != 1.0:
            continue
        atom = atomic_num_at(data, nbr)
        if atom in (7, 8, 16, 34):
            heteros.append(nbr)
    return heteros


def match_acyl_from_fragment(data, attachment, neighbor, frag_key):

    if frag_key in ("COOMe", "COOMe_inv", "CONH2", "CONH2_inv"):
        carbonyl = None
        adj = adjacency_list(data.edge_index, data.x.size(0))
        if frag_key in ("COOMe", "CONH2"):
            # attachment -> carbonyl
            if atomic_num_at(
                data, neighbor) == 6 and bond_order_between(
                data, attachment, neighbor
            ) == 1.0:
                carbonyl = neighbor
        else:
            if bond_order_between(
                data, attachment, neighbor) == 1.0:
                for nbr in adj[neighbor]:
                    if nbr == attachment:
                        continue
                    if atomic_num_at(
                        data, nbr) == 6 and bond_order_between(
                        data, neighbor, nbr) == 1.0:
                        carbonyl = nbr
                        break
        if carbonyl is not None:
            heteros = _carbonyl_single_heteros(
                data, carbonyl, exclude=neighbor
                )
            if heteros:
                return False
    if match_fragment(data, attachment, neighbor, frag_key):
        return True
    if frag_key == "COOMe":
        return match_from_carbonyl(
            data, attachment, neighbor
            )
    if frag_key == "COOMe_inv":
        return match_from_oxygen(
            data, attachment, neighbor
            )
    if frag_key == "thioester":
        return match_carbonyl(
            data, attachment, neighbor
            )
    if frag_key == "thioester_inv":
        return match_from_sulfur(
            data, attachment, neighbor
            )
    if frag_key == "CONH2":
        return match_amide_from_carbonyl(
            data, attachment, neighbor
            )
    if frag_key == "CONH2_inv":
        return match_from_nitrogen(
            data, attachment, neighbor
            )
    
    return False


def violates_blacklist(data):
    hetero = {7, 8, 15, 16, 34}
    halogens = {9, 17, 35, 53}
    if data.edge_attr is None or data.edge_attr.numel() == 0:
        return False
    def is_nitro_like(n_idx):
        if atomic_num_at(data, n_idx) != 7:
            return False
        adj = adjacency_list(
            data.edge_index, data.x.size(0))
        o_neighbors = [n for n in adj[
            n_idx] if atomic_num_at(data, n) == 8]
        if len(o_neighbors) != 2:
            return False
        if atomic_charge_at(data, n_idx) == 1:
            neg_oxos = sum(1 for n in o_neighbors 
                    if atomic_charge_at(data, n) == -1)
            return neg_oxos >= 1
        return False
    src, dst = data.edge_index
    orders = bond_orders(data.edge_attr)
    for e_idx, (i, j) in enumerate(
        zip(src.tolist(), dst.tolist())):
        ai = atomic_num_at(data, i)
        aj = atomic_num_at(data, j)
        order = float(orders[e_idx].item())
        if order == 1.0 and ((
            ai == 7 and aj == 8) or (ai == 8 and aj == 7)):
            n = i if ai == 7 else j
            if is_nitro_like(n):
                continue
            mask = (src == n) | (dst == n)
            has_oxo = False
            for k in torch.where(mask)[0].tolist():
                ni = int(dst[k].item()) if int(
                    src[k].item()) == n else int(src[k].item())
                if atomic_num_at(data, ni) == 8 and float(
                    orders[k].item()) == 2.0:
                    has_oxo = True
                    break
            if not has_oxo:
                return True
        if ai in hetero and aj in hetero and order == 1.0:
            if (ai == 7 and aj == 8 and is_nitro_like(i)):
                continue
            if (aj == 7 and ai == 8 and is_nitro_like(j)):
                continue
            return True
        if ai == 8 and aj == 8 and order == 1.0:
            return True
        if (ai == 8 and aj in halogens) or (
            aj == 8 and ai in halogens):
            return True
        if (ai == 7 and aj == 7 and order == 1.0):
            return True
        if (ai in halogens and aj in hetero) or (
            aj in halogens and ai in hetero):
            return True
        if (ai == 6 and aj in halogens) or (
            aj == 6 and ai in halogens):
            c = i if ai == 6 else j
            mask = (src == c) | (dst == c)
            has_o_double = False
            for k in torch.where(mask)[0].tolist():
                ni = int(dst[k].item()) if int(
                    src[k].item()) == c else int(src[k].item())
                if atomic_num_at(data, ni) == 8 and float(
                    orders[k].item()) == 2.0:
                    has_o_double = True
                    break
            if has_o_double:
                return True
    return False


def valence_ok(atom_num, valence, in_ring=False):
    max_val = MAX_VALENCE.get(atom_num, 4)
    if in_ring:
        max_val = min(max_val, 
            RING_MAX_VALENCE.get(atom_num, max_val))
        
    return valence <= max_val + 1e-6


def rule_fragments(direction):
    if "->" not in direction:
        return None, None
    left, right = direction.split("->", 1)
    
    return left.strip(), right.strip()


def sp2_polar_filters(data, attachment, neighbor, left, right):
    if not is_sp2_attachment(data, attachment):
        return False
    if adjacent_to_carbonyl(data, attachment
            ) or adjacent_to_carbonyl(data, neighbor):
        return False
    if left in ("OH", "OMe", "SH"
            ) and is_acyl_hetero_link(data, attachment, neighbor):
        return False
    if left == "F" and is_cf3_f(data, neighbor):
        return False
    if right in ("F", "Cl", "Br", "I"):
        if is_amidine_core(data, attachment
            ) or is_amidine_core(data, neighbor):
            return False
        
    return True


def sp2_polar_rules():
    rules = []
    for left in POLAR_FRAGMENTS:
        for right in POLAR_FRAGMENTS:
            if left == right:
                continue
            rule_id = f"SP2_POLAR_ALL_{left}_TO_{right}"
            rules.append(
                {"rule_id": rule_id,
                "family": "SP2_POLAR_ALL",
                "direction": f"{left}->{right}",
                "attachment_ctx": "sp2_attachment"}
                )
    return rules


def sp2_apolar_rules():
    rules = []
    for left in APOLAR_FRAGMENTS:
        for right in APOLAR_FRAGMENTS:
            if left == right:
                continue
            rule_id = f"SP2_APOLAR_ALL_{left}_TO_{right}"
            rules.append(
                {"rule_id": rule_id,
                "family": "SP2_APOLAR_ALL",
                "direction": f"{left}->{right}",
                "attachment_ctx": "sp2_attachment"}
                )
    return rules


def sp3_polar_rules():
    rules = []
    for left in POLAR_FRAGMENTS:
        for right in POLAR_FRAGMENTS:
            if left == right:
                continue
            rule_id = f"SP3_POLAR_ALL_{left}_TO_{right}"
            rules.append(
                {"rule_id": rule_id,
                "family": "SP3_POLAR_ALL",
                "direction": f"{left}->{right}",
                "attachment_ctx": "sp3_attachment"}
                )
    return rules


def sp3_apolar_rules():
    rules = []
    for left in APOLAR_FRAGMENTS:
        for right in APOLAR_FRAGMENTS:
            if left == right:
                continue
            rule_id = f"SP3_APOLAR_ALL_{left}_TO_{right}"
            rules.append(
                {"rule_id": rule_id,
                "family": "SP3_APOLAR_ALL",
                "direction": f"{left}->{right}",
                "attachment_ctx": "sp3_attachment"}
                )
    return rules


def sp3_reactive_rules():
    rules = []
    for left in REACTIVE_FRAGMENTS:
        for right in REACTIVE_FRAGMENTS:
            if left == right:
                continue
            rule_id = f"SP3_REACTIVE_ALL_{left}_TO_{right}"
            rules.append(
                {"rule_id": rule_id,
                "family": "SP3_REACTIVE_ALL",
                "direction": f"{left}->{right}",
                "attachment_ctx": "sp3_attachment"}
                )
    return rules


def sp2_reactive_rules():
    rules = []
    for left in REACTIVE_FRAGMENTS:
        for right in REACTIVE_FRAGMENTS:
            if left == right:
                continue
            rule_id = f"SP2_REACTIVE_ALL_{left}_TO_{right}"
            rules.append(
                {"rule_id": rule_id,
                "family": "SP2_REACTIVE_ALL",
                "direction": f"{left}->{right}",
                "attachment_ctx": "sp2_attachment"}
                )
    return rules


def acyl_family_rules():
    rules = []
    for left in ACYL_FAMILY_FRAGMENTS:
        for right in ACYL_FAMILY_FRAGMENTS:
            if left == right:
                continue
            if (left == "COOH" and right == "COOMe") or (
                left == "COOMe" and right == "COOH"):
                continue
            rule_id = f"ACYL_FAMILY_ALL_{left}_TO_{right}"
            rules.append(
                {"rule_id": rule_id,
                "family": "ACYL_FAMILY_ALL",
                "direction": f"{left}->{right}",
                "attachment_ctx": "sp2_sp3_attachment"}
                )
    return rules


def amide_family_rules():
    rules = []
    for left in AMIDE_FAMILY_FRAGMENTS:
        for right in AMIDE_FAMILY_FRAGMENTS:
            if left == right:
                continue
            rule_id = f"AMIDE_FAMILY_ALL_{left}_TO_{right}"
            rules.append(
                {"rule_id": rule_id,
                "family": "AMIDE_FAMILY_ALL",
                "direction": f"{left}->{right}",
                "attachment_ctx": "sp2_sp3_attachment"}
                )
    return rules


def redox_family_rules():
    rules = []
    directions = [
        "alcohol->aldehyde",
        "aldehyde->alcohol",
        "secondary_alcohol->ketone",
        "ketone->alcohol",
        "thiol->thio",
        "thio->thiol",
        "selenol->seleno",
        "seleno->selenol"
        ]
    for direction in directions:
        rule_id = f"REDOX_FAMILY_{direction}"
        rules.append(
            {"rule_id": rule_id,
            "family": "REDOX_FAMILY",
            "direction": direction,
            "attachment_ctx": "redox_core"}
            )
    return rules


def toggle_ring_family_rules():
    return [
        {"rule_id": "TOGGLE_AROMATIC",
        "family": "TOGGLE_RING_FAMILY_ALL",
        "direction": "toggle_ring_aromatic",
        "attachment_ctx": "ring_atom"}
        ]


def ring_family_rules():
    return [
        {"rule_id": "RING_FAMILY_ALL_ISOSTERIC",
        "family": "RING_FAMILY_ALL",
        "direction": "ring_isostere",
        "attachment_ctx": "ring_atom"}
        ]


def bond_family_rules():
    return [
        {"rule_id": "BOND_FAMILY_ALL_PERTURB",
        "family": "BOND_FAMILY_ALL",
        "direction": "perturb_bond",
        "attachment_ctx": "bond_atom"}
        ]


def diaryl_family_rules():
    return [
        {"rule_id": "DIARYL_FAMILY_ALL_SWAP",
        "family": "DIARYL_FAMILY_ALL",
        "direction": "swap_diatomic",
        "attachment_ctx": "bond_atom"}
        ]


def toggle_charge_family_rules():
    return [
        {"rule_id": "TOGGLE_CHARGE_FAMILY_ALL",
        "family": "TOGGLE_CHARGE_FAMILY_ALL",
        "direction": "toggle_charge",
        "attachment_ctx": "charge_atom"}
        ]


def polyvalent_family_rules():
    return [
        {"rule_id": "POLYVALENT_FAMILY_ALL_SWAP",
        "family": "POLYVALENT_FAMILY_ALL",
        "direction": "polyvalent_swap",
        "attachment_ctx": "center_atom"}
        ]


def aliphatic_family_rules():
    return [
        {"rule_id": "ALIPHATIC_FAMILY_ALL_ISOSTERIC",
        "family": "ALIPHATIC_FAMILY_ALL",
        "direction": "aliphatic_isostere",
        "attachment_ctx": "aliphatic_atom"}
        ]


def carbamate_family_rules():
    rules = []
    for left in CARBAMATE_FAMILY_FRAGMENTS:
        for right in CARBAMATE_FAMILY_FRAGMENTS:
            if left == right:
                continue
            rule_id = f"CARBAMATE_FAMILY_ALL_{left}_TO_{right}"
            rules.append(
                {"rule_id": rule_id,
                "family": "CARBAMATE_FAMILY_ALL",
                "direction": f"{left}->{right}",
                "attachment_ctx": "sp2_sp3_attachment"}
                )
    return rules


def sulfure_family_rules():
    rules = []
    for left in SULFONE_SULFONAMIDE_FRAGMENTS:
        for right in SULFONE_SULFONAMIDE_FRAGMENTS:
            if left == right:
                continue
            rule_id = f"SULFURE_FAMILY_ALL_{left}_TO_{right}"
            rules.append(
                {"rule_id": rule_id,
                "family": "SULFURE_FAMILY_ALL",
                "direction": f"{left}->{right}",
                "attachment_ctx": "sp2_sp3_attachment"}
                )
    return rules