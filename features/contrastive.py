import copy
import functools
import math
import random

import networkx as nx
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from rdkit import Chem
from rdkit.Chem.BRICS import (
    BRICSDecompose,
    BreakBRICSBonds,
    FindBRICSBonds,
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


def _batch_graphs(graphs):
    batch = Batch.from_data_list(graphs)
    batch.smiles = [g.smiles for g in graphs]
    labels = [getattr(g, "y", None) for g in graphs]
    batch.y = _collate_labels(labels)
    return batch


def build_mask_token(node_feature_dim):
    return torch.zeros(node_feature_dim, dtype=torch.float32)


def augment_graph_view(
    data,
    node_mask_ratio=0.25,
    edge_mask_ratio=0.25,
    mask_token=None,
):
    augmented = copy.deepcopy(data)
    num_nodes = int(augmented.x.size(0))
    num_edges = int(augmented.edge_index.size(1))

    if mask_token is None:
        mask_token = build_mask_token(augmented.x.size(1))
    mask_token = mask_token.to(
        device=augmented.x.device,
        dtype=augmented.x.dtype,
    )

    num_mask_nodes = 0
    if num_nodes > 0 and node_mask_ratio > 0:
        num_mask_nodes = max(1, math.floor(node_mask_ratio * num_nodes))
        num_mask_nodes = min(num_mask_nodes, num_nodes)

    if num_mask_nodes > 0:
        node_indices = random.sample(range(num_nodes), num_mask_nodes)
        augmented.x[node_indices] = mask_token

    num_mask_edges = 0
    if num_edges > 0 and edge_mask_ratio > 0:
        num_mask_edges = max(1, math.floor(edge_mask_ratio * num_edges))
        num_mask_edges = min(num_mask_edges, num_edges)

    if num_mask_edges > 0:
        edge_indices = set(random.sample(range(num_edges), num_mask_edges))
        keep_mask = torch.tensor(
            [idx not in edge_indices for idx in range(num_edges)],
            dtype=torch.bool,
            device=augmented.edge_index.device,
        )
        augmented.edge_index = augmented.edge_index[:, keep_mask].clone()
        augmented.edge_attr = augmented.edge_attr[keep_mask].clone()

    return augmented


def _reference_fragment_indices(mol):
    graph = nx.Graph()
    graph.add_nodes_from(range(mol.GetNumAtoms()))
    graph.add_edges_from(
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in mol.GetBonds()
    )

    brics_bonds = list(FindBRICSBonds(mol))
    break_bonds = [bond_info[0] for bond_info in brics_bonds]
    graph.remove_edges_from(break_bonds)

    components = []
    for component in nx.connected_components(graph):
        if len(component) > 3:
            components.append(tuple(sorted(component)))
    return set(components)


def get_fragment_atom_sets(mol, min_size=4):
    if mol is None:
        return []

    try:
        reference_sets = _reference_fragment_indices(mol)
        fragments = list(BRICSDecompose(mol, returnMols=True))
        broken = BreakBRICSBonds(mol)

        dummy_atoms = {
            atom.GetIdx()
            for atom in broken.GetAtoms()
            if atom.GetAtomicNum() == 0
        }

        atom_sets = []
        seen = set()
        for fragment in fragments:
            matches = broken.GetSubstructMatches(fragment)
            for match in matches:
                atom_idx = tuple(sorted(set(match) - dummy_atoms))
                if len(atom_idx) < min_size:
                    continue
                if len(matches) > 1 and atom_idx not in reference_sets:
                    continue
                if atom_idx in seen:
                    continue
                seen.add(atom_idx)
                atom_sets.append(atom_idx)
        return atom_sets
    except Exception:
        return []


@functools.lru_cache(maxsize=50000)
def get_cached_fragment_atom_sets(smiles, min_size=4):
    if smiles is None:
        return tuple()
    mol = Chem.MolFromSmiles(smiles)
    fragment_atoms = get_fragment_atom_sets(
        mol,
        min_size=min_size,
    )
    return tuple(tuple(atom_set) for atom_set in fragment_atoms)


class ContrastiveGraphDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        node_mask_ratio=0.25,
        edge_mask_ratio=0.25,
        min_fragment_size=4,
    ):
        self.base_dataset = base_dataset
        self.node_mask_ratio = node_mask_ratio
        self.edge_mask_ratio = edge_mask_ratio
        self.min_fragment_size = min_fragment_size

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        graph = self.base_dataset[idx]
        if graph is None:
            return None

        original = copy.deepcopy(graph)
        view_i = augment_graph_view(
            original,
            node_mask_ratio=self.node_mask_ratio,
            edge_mask_ratio=self.edge_mask_ratio,
        )
        view_j = augment_graph_view(
            original,
            node_mask_ratio=self.node_mask_ratio,
            edge_mask_ratio=self.edge_mask_ratio,
        )

        smiles = getattr(original, "smiles", None)
        fragment_atoms = get_cached_fragment_atom_sets(
            smiles,
            min_size=self.min_fragment_size,
        )

        return {
            "supervised": original,
            "view_i": view_i,
            "view_j": view_j,
            "smiles": smiles,
            "fragment_atoms": fragment_atoms,
        }


def collate_contrastive_graphs(samples):
    samples = [
        sample for sample in samples
        if sample is not None and sample.get("supervised") is not None
    ]
    if not samples:
        return None

    originals = [sample["supervised"] for sample in samples]
    view_i = [sample["view_i"] for sample in samples]
    view_j = [sample["view_j"] for sample in samples]

    supervised_batch = _batch_graphs(originals)
    view_i_batch = _batch_graphs(view_i)
    view_j_batch = _batch_graphs(view_j)

    atom_membership = []
    fragment_membership = []
    fragment_batch = []
    fragment_counter = 0
    node_offset = 0

    for graph_idx, sample in enumerate(samples):
        for atom_set in sample["fragment_atoms"]:
            global_atom_idx = [
                node_offset + atom_idx
                for atom_idx in atom_set
            ]
            atom_membership.extend(global_atom_idx)
            fragment_membership.extend(
                [fragment_counter] * len(global_atom_idx)
            )
            fragment_batch.append(graph_idx)
            fragment_counter += 1
        node_offset += int(sample["supervised"].x.size(0))

    if atom_membership:
        fragment_atom_index = torch.tensor(
            atom_membership,
            dtype=torch.long,
        )
        fragment_index = torch.tensor(
            fragment_membership,
            dtype=torch.long,
        )
        fragment_batch = torch.tensor(
            fragment_batch,
            dtype=torch.long,
        )
    else:
        fragment_atom_index = torch.zeros(0, dtype=torch.long)
        fragment_index = torch.zeros(0, dtype=torch.long)
        fragment_batch = torch.zeros(0, dtype=torch.long)

    return {
        "supervised": supervised_batch,
        "view_i": view_i_batch,
        "view_j": view_j_batch,
        "smiles": [sample["smiles"] for sample in samples],
        "fragment_atom_index": fragment_atom_index,
        "fragment_index": fragment_index,
        "fragment_batch": fragment_batch,
        "num_fragments": fragment_counter,
    }
