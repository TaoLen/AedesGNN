import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def calc_fp(
    SMILES, 
    fp_size=2048, 
    radius=2):

    mol = Chem.MolFromSmiles(SMILES, sanitize=True)
    if mol is None:
        return np.zeros((fp_size,), dtype=np.float32)
    mol.UpdatePropertyCache(False)
    Chem.GetSSSR(mol)
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=fp_size
        )
    fp = mfpgen.GetFingerprint(mol)
    arr = np.zeros((fp_size,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr


def assign_fp(
    smiles,
    fp_size: int = 2048,
    radius: int = 2):

    valid_smiles = []
    invalid_smiles = []

    for smi in smiles:
        if Chem.MolFromSmiles(smi):
            valid_smiles.append(smi)
        else:
            invalid_smiles.append(smi)
    descs = [calc_fp(smi, fp_size=fp_size, radius=radius)
             for smi in valid_smiles]
    descs = [d for d in descs if d.shape == (fp_size,)]
    descs = np.asarray(descs, dtype=np.float32)

    return descs, invalid_smiles


class Bit2Dataset(Dataset):
    def __init__(self, fingerprints, labels):
        self.fingerprints = fingerprints
        self.labels = labels

    def __len__(self):
        return len(self.fingerprints)

    def __getitem__(self, idx):
        fp = self.fingerprints[idx]
        lbl = self.labels[idx]
        return (
            torch.tensor(fp, dtype=torch.float32),
            torch.tensor(lbl, dtype=torch.float32)
            )
