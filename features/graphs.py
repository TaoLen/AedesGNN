import torch
from torch.utils.data import Dataset
from rdkit import Chem
from torch_geometric.data import Data
from utils import one_hot


def ATOMIC_NUMBER():
    return [
        # Alkali Metals (1 electron in the valence shell)
        1, 11, 19,
        # Alkaline Earth Metals (2 electrons in the valence shell)
        12, 20,
        # Transition Metals (various numbers of electrons)
        22, 23, 24, 25, 26, 27, 28, 29, 30,
        # Metalloids and elements with 3-5 electrons in the valence shell
        5, 13, 14, 15, 33,
        # Non-metals and elements with 6-7 electrons in the valence shell
        6, 7, 8, 16, 34, 9, 17,
        # Other halogens
        35, 53
        ]

def get_atomic_number():
    atomic_number_list = ATOMIC_NUMBER()
    def atomic_number(number):
        return one_hot(number, atomic_number_list)
    return atomic_number


def get_degree(degree):
    degree_categories = list(range(0, 8))
    return one_hot(degree, degree_categories)


def get_charge(charge):
    charge_categories = [-1, 0, 1, 2, 3, 4]
    return one_hot(charge, charge_categories)


def get_hybridization(hybridization):
    hybridization_categories = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
        ]
    if hybridization not in hybridization_categories:
        hybridization = Chem.rdchem.HybridizationType.SP3

    return one_hot(
        hybridization, 
        hybridization_categories
        )


def get_chirality(atom):

    chiral_tag = atom.GetChiralTag()
    if chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
        return 'R'
    elif chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
        return 'S'
    return None


def encode_stereo(chirality, stereo):
    bond_stereo_categories = [
        Chem.rdchem.BondStereo.STEREONONE, 
        Chem.rdchem.BondStereo.STEREOANY, 
        Chem.rdchem.BondStereo.STEREOZ,  
        Chem.rdchem.BondStereo.STEREOE, 
        'R', 'S']
    
    return one_hot(chirality if chirality 
                   else stereo, 
                   bond_stereo_categories
                   )


def get_bond_stereo(stereo, atom=None):
    chirality = get_chirality(atom
            ) if atom else None

    return encode_stereo(chirality, stereo)


def atom_features(atom):
    atomic_number = get_atomic_number()
    atom_feature = torch.cat([
        torch.tensor(atomic_number(
            atom.GetAtomicNum()), 
            dtype=torch.float),
        torch.tensor(get_degree(
            atom.GetDegree()), 
            dtype=torch.float),
        torch.tensor(get_charge(
            atom.GetFormalCharge()), 
            dtype=torch.float),
        torch.tensor(get_hybridization(
            atom.GetHybridization()), 
            dtype=torch.float),
        torch.tensor([float(
            atom.GetIsAromatic())], 
            dtype=torch.float)
            ]
        )
    return atom_feature


def bond_features(bond):
    bond_feature = torch.cat([
        torch.tensor(one_hot(bond.GetBondType(), [
            Chem.rdchem.BondType.SINGLE, 
            Chem.rdchem.BondType.DOUBLE, 
            Chem.rdchem.BondType.TRIPLE, 
            Chem.rdchem.BondType.AROMATIC]), 
            dtype=torch.float),
        torch.tensor([float(
            bond.GetIsConjugated())], 
            dtype=torch.float),
        torch.tensor([float(
            bond.IsInRing())], 
            dtype=torch.float),
        torch.tensor(
            get_bond_stereo(bond.GetStereo()), 
            dtype=torch.float)
            ]
        )
    return bond_feature


def smiles2graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atom_features_list = [
        atom_features(atom) 
        for atom in mol.GetAtoms()
        ]
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feature = bond_features(bond)
        edge_indices.append([i, j])
        edge_features.append(bond_feature)
    x = torch.stack(atom_features_list)
    edge_index = torch.tensor(
        edge_indices, 
        dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_features)

    return Data(x=x, edge_index=edge_index, 
                edge_attr=edge_attr
               )


class graph2dataset(Dataset):
    def __init__(self, smiles_list, labels=None):
        self.smiles_list = smiles_list
        self.labels = labels
    def __len__(self):
        return len(self.smiles_list)
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        label = self.labels[idx] if self.labels is not None else None
        data = smiles2graph(smiles)
        data.smiles = smiles
        if data is None:
            return None
        if label is not None:
            data.y = torch.tensor(
                label, dtype=torch.float32).unsqueeze(0)
            
        return data