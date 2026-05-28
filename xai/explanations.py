import os
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
from torch_geometric.data import Data
from torch_geometric.utils import (
    get_laplacian,
    to_dense_adj
    )

from weights import (
    zscore_norm,
    extract_node_weights,
    extract_bond_vecs
    )

from counterfactuals import leave_one_group


def plot_contours(
    index, smiles, 
    importance, 
    out_path):

    DrawingOptions.atomLabelFontSize = 35
    DrawingOptions.dotsPerAngstrom = 60
    DrawingOptions.useBWAtomPalette = True

    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    rdDepictor.StraightenDepiction(mol)

    fig = SimilarityMaps.GetSimilarityMapFromWeights(
        mol, importance,
        alpha=0.5, 
        contourLines=6,
        colorMap='bwr', 
        size=(200, 200)
        )
    fig.patch.set_facecolor('white')
    for ax in fig.axes:
        ax.set_facecolor('white')
        ax.axis('off')
        for line in ax.get_lines():
            line.set_color('black')
            line.set_linewidth(2)
        for coll in ax.collections:
            coll.set_edgecolor('black')
            coll.set_facecolor(coll.get_facecolor())
        for txt in ax.texts:
            txt.set_color('black')
            txt.set_fontweight('bold')
            txt.set_path_effects([
                pe.Stroke(
                    linewidth=5,
                    foreground='white'),
                pe.Normal()
                ]
            )
    os.makedirs(out_path, exist_ok=True)
    fig.savefig(
        f"{out_path}/{index}.svg",
        bbox_inches='tight',
        pad_inches=0.6,
        dpi=300,
        facecolor='white'
        )
    
    plt.close(fig)


def view_explanations(
    model, 
    loader, 
    device, 
    out_path,
    num_perturb=10,
    task_idx=0):

    model.to(device)
    model.eval()
    index = 1
    with torch.no_grad():
        for batch in loader:
            raw = copy.deepcopy(batch)
            smiles = raw.smiles
            batch = batch.to(device)
            bi = batch.batch
            src, dst = batch.edge_index
            counts = torch.bincount(bi)
            counts = counts.cpu().numpy()
            start = 0

            for i, c in enumerate(counts):
                n = int(c)
                mask = (bi[src] == i) & (bi[dst] == i)
                ei = batch.edge_index[:, mask].cpu().clone()
                ei[0] -= start
                ei[1] -= start
                ea = batch.edge_attr[mask].cpu().clone()

                data = Data(
                    x=raw.x[start:start+n].cpu().clone(),
                    edge_index=ei,
                    edge_attr=ea,
                    smiles=smiles[i]
                    )
                mean, _, pos, neg = leave_one_group(
                    model, data, device,
                    num_perturb=num_perturb,
                    task_idx=task_idx
                    )
                weight = pos - neg
                node_imp = mean * weight
                imp = node_imp.copy()
                nz_mask = np.abs(imp) > 1e-12
                if np.any(nz_mask):
                    nz_mean = imp[nz_mask].mean()
                    nz_std = imp[nz_mask].std()
                    if nz_std > 1e-12:
                        imp[nz_mask] = (
                            imp[nz_mask] - nz_mean) / (nz_std + 1e-8)
                    imp[~nz_mask] = 0.0
                plot_contours(
                    index, data.smiles,
                    imp, out_path
                    )
                
                start += n
                index += 1


def plot_attentions(
    out_path, 
    index, 
    smiles, 
    attn,
    prefix=None):

    DrawingOptions.atomLabelFontSize = 35
    DrawingOptions.dotsPerAngstrom = 60
    DrawingOptions.useBWAtomPalette = True
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    rdDepictor.StraightenDepiction(mol)
    fig = SimilarityMaps.GetSimilarityMapFromWeights(
        mol, attn,
        alpha=0.5,
        contourLines=6,
        colorMap='bwr',
        size=(200, 200)
        )
    fig.patch.set_facecolor('white')
    for ax in fig.axes:
        ax.set_facecolor('white')
        ax.axis('off')
        for line in ax.get_lines():
            line.set_color('black')
            line.set_linewidth(2)
        for coll in ax.collections:
            coll.set_edgecolor('black')
            coll.set_facecolor(
                coll.get_facecolor()
                )
        for txt in ax.texts:
            txt.set_color('black')
            txt.set_fontweight('bold')
            txt.set_path_effects([
                pe.Stroke(
                    linewidth=5,
                    foreground='white'),
                pe.Normal()
                ]
            )
    os.makedirs(out_path, exist_ok=True)
    name = f"{index}.svg" if prefix is None else f"{prefix}_{index}.svg"
    fig.savefig(
        f"{out_path}/{name}",
        bbox_inches='tight',
        pad_inches=0.6,
        dpi=300,
        facecolor='white'
        )
    
    plt.close(fig)


def view_attentions(
    model, 
    loader, 
    device, 
    out_path,
    arm="node"):

    model.to(device)
    model.eval()
    index = 1
    arm = "node" if arm is None else str(arm).lower().strip()
    if arm not in {"node", "edge"}:
        raise ValueError("arm must be 'node' or 'edge'")

    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(device)
            smiles = batch_data.smiles
            x = batch_data.x
            edge_index = batch_data.edge_index
            edge_attr = batch_data.edge_attr
            batch_idx = batch_data.batch

            if arm == "node" and (
                hasattr(model, "node_readout")
                or hasattr(model, "atom_attention")
                or hasattr(model, "node_agg_layers")
            ):
                attn = extract_node_weights(
                    model, batch_data, device)
            elif arm == "edge" and (
                hasattr(model, "agg_layers")
                or hasattr(model, "edge_agg_layers")
            ):
                attn = extract_bond_vecs(
                    model, batch_data, device)
            else:
                for layer in model.agg_layers:
                    try:
                        out = layer(x, edge_index, edge_attr, batch_idx)
                    except TypeError:
                        try:
                            out = layer(x, edge_index, edge_attr)
                        except TypeError:
                            out = layer(x, edge_index)
                    x = out[0] if isinstance(out, tuple) else out
                lap_idx, lap_w = get_laplacian(
                    edge_index,
                    edge_weight=torch.ones(
                        edge_index.size(1), device=device),
                    normalization='sym'
                    )
                laplacian_matrix = to_dense_adj(
                    lap_idx, edge_attr=lap_w,
                    max_num_nodes=x.size(0))[0]
                adjacency_matrix = torch.eye(
                    x.size(0), device=device
                    )
                if hasattr(model, 'atom_attention'):
                    att_out, attn = model.atom_attention(
                        x,
                        edge_index,
                        batch_idx,
                        laplacian_matrix,
                        adjacency_matrix
                        )
                else:
                    last_layer = model.agg_layers[-1]
                    alpha = getattr(last_layer, "last_attn_weights", None)
                    if alpha is None:
                        raise AttributeError(
                            "view_attentions requires attention"
                            )
                    N = x.size(0)
                    att = torch.zeros(N, N, device=device)
                    src, dst = edge_index
                    att[src, dst] = alpha
                    attn = (att + att.t()) * 0.5

            attn = attn.detach().cpu()
            counts = torch.bincount(batch_idx).cpu().numpy()
            start = 0

            for i, c in enumerate(counts):
                n = int(c)
                if attn.dim() == 2:
                    sub_attn = attn[start:start + n, start:start + n]
                    att_vals = (sub_attn.sum(0).cpu().numpy() / max(n, 1))
                else:
                    att_vals = attn[start:start + n].cpu().numpy()
                att_vals = zscore_norm(att_vals)
                out_dir = os.path.join(out_path, arm)
                plot_attentions(
                    out_dir, index, smiles[i], 
                    att_vals, prefix=arm
                    )

                start += n
                index += 1
