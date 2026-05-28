import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FixedLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utils import task_inference


_DROPOUT_TYPES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)

_INACTIVE_COLOR = "#756CF4"
_ACTIVE_COLOR = "#FC7777"
_UNLABELED_COLOR = "#9E9E9E"


def split_model_outputs(outputs):
    outputs_scalar = None
    mc_logits = None
    if isinstance(outputs, dict):
        outputs_scalar = outputs.get("scalar")
        mc_logits = outputs.get("mc_logits", {})
    else:
        if outputs.dim() == 3:
            mc_logits = outputs
            outputs_scalar = outputs[..., 0]
        else:
            outputs_scalar = outputs
    if outputs_scalar is None:
        raise ValueError("Model must return scalar logits.")
    return outputs_scalar, mc_logits


def _binary_entropy(probs, eps=1e-12):
    p = probs.clamp(min=eps, max=1.0 - eps)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


def categorical_entropy(probs, dim=-1, eps=1e-12):
    p = probs.clamp(min=eps, max=1.0)
    return -(p * p.log()).sum(dim=dim)


def _resolve_task_type_for_batch(
    model, task_type, y_true, outputs_scalar):
    tt = task_type
    if tt is None:
        tt = getattr(model, "task_type", None)
    if tt is None:
        if y_true is None:
            raise ValueError(
                "task_type is required for unlabeled external data."
            )
        mask = ~torch.isnan(y_true)
        y_true_z = torch.nan_to_num(y_true, nan=0.0)
        tt = task_inference(
            y_true_z, mask
            ).to(outputs_scalar.device)
    elif not torch.is_tensor(tt):
        tt = torch.tensor(
            tt, device=outputs_scalar.device, dtype=torch.long
            )
    else:
        tt = tt.to(outputs_scalar.device)
    return tt


def activate_dropout(model):
    prev_mode = model.training
    model.eval()
    dropout_layers = []
    for module in model.modules():
        if isinstance(module, _DROPOUT_TYPES):
            dropout_layers.append((module, module.training))
            module.train(True)
    return prev_mode, dropout_layers


def restore_dropout(model, prev_mode, dropout_layers):
    for module, was_training in dropout_layers:
        module.train(was_training)
    model.train(prev_mode)


def init_batch_uncertainty(batch_size, num_tasks, device, dtype):
    nan_mat = torch.full(
        (batch_size, num_tasks),
        float("nan"),
        device=device,
        dtype=dtype
        )
    return {
        "pred_mean": nan_mat.clone(),
        "pred_var": nan_mat.clone(),
        "pred_std": nan_mat.clone(),
        "bin_var": nan_mat.clone(),
        "bin_entropy": nan_mat.clone(),
        "bin_mutual_info": nan_mat.clone(),
        "bin_threshold": nan_mat.clone(),
        "bin_decision_margin": nan_mat.clone(),
        "bin_confidence_margin": nan_mat.clone(),
        "bin_mc_positive_rate": nan_mat.clone(),
        "bin_threshold_instability": nan_mat.clone(),
        "reg_var": nan_mat.clone(),
        "reg_std": nan_mat.clone(),
        "mc_entropy": nan_mat.clone(),
        "mc_expected_entropy": nan_mat.clone(),
        "mc_mutual_info": nan_mat.clone(),
        "mc_variation_ratio": nan_mat.clone(),
        }


def _stack_or_none(items):
    if not items:
        return None
    return np.concatenate(items, axis=0)


def _batch_smiles(batch, batch_size):
    smiles = getattr(batch, "smiles", None)
    if smiles is None:
        return [None] * int(batch_size)
    if isinstance(smiles, (list, tuple)):
        vals = [None if s is None else str(s) for s in smiles]
        if len(vals) < int(batch_size):
            vals.extend([None] * (int(batch_size) - len(vals)))
        return vals[:int(batch_size)]
    vals = [str(smiles)]
    if int(batch_size) > 1:
        vals.extend([None] * (int(batch_size) - 1))
    return vals


def _load_thresholds(thresholds_path, thresholds_split="val"):
    if thresholds_path is None:
        return None
    with open(thresholds_path, "r") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        if thresholds_split in payload:
            return payload[thresholds_split]
        if "thresholds" in payload:
            return payload["thresholds"]
    return payload


def _resolve_thresholds(
    thresholds,
    num_tasks,
    device=None,
    dtype=torch.float32):

    if thresholds is None:
        values = np.full(int(num_tasks), 0.5, dtype=float)
    elif isinstance(thresholds, dict):
        values = np.full(int(num_tasks), 0.5, dtype=float)
        for key, value in thresholds.items():
            try:
                idx = int(key)
            except Exception:
                continue
            if 0 <= idx < int(num_tasks):
                values[idx] = float(value)
    else:
        values = np.asarray(thresholds, dtype=float)
        if values.ndim == 0:
            values = np.full(int(num_tasks), float(values), dtype=float)
        if values.size < int(num_tasks):
            padded = np.full(int(num_tasks), 0.5, dtype=float)
            padded[:values.size] = values
            values = padded
        else:
            values = values[:int(num_tasks)]
    values = np.clip(values, 1e-6, 1.0 - 1e-6)
    tensor = torch.tensor(values, dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def predict_uncertainty(
    model,
    data_loader,
    device,
    n_samples=30,
    task_type=None,
    mc_label_values=None,
    mc_labels=None,
    n_mc_samples=None,
    use_thresholds=False,
    thresholds_path="../output/calibration/thresholds.json",
    thresholds_split="val",
    thresholds=None):
    if n_mc_samples is not None:
        n_samples = n_mc_samples
    if mc_label_values is None and mc_labels is not None:
        mc_label_values = mc_labels
    if thresholds is None and use_thresholds:
        thresholds = _load_thresholds(
            thresholds_path,
            thresholds_split=thresholds_split,
        )
    model = model.to(device)
    n_samples = int(n_samples)
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2.")
    labels = []
    pred_mean_all = []
    pred_var_all = []
    pred_std_all = []
    bin_var_all = []
    bin_entropy_all = []
    bin_mi_all = []
    bin_threshold_all = []
    bin_decision_margin_all = []
    bin_confidence_margin_all = []
    bin_mc_positive_rate_all = []
    bin_threshold_instability_all = []
    reg_var_all = []
    reg_std_all = []
    mc_entropy_all = []
    mc_expected_entropy_all = []
    mc_mi_all = []
    mc_vr_all = []
    mc_prob_mean = {}
    smiles_all = []

    prev_mode, dropout_layers = activate_dropout(model)
    try:
        with torch.no_grad():
            for batch in data_loader:
                if batch is None:
                    continue
                inputs = batch.to(device)
                y_true = None
                if hasattr(inputs, "y") and inputs.y is not None:
                    y_true = inputs.y
                    labels.append(y_true.detach().cpu().numpy())

                scalar_samples = []
                mc_logits_samples = {}
                for _ in range(n_samples):
                    outputs = model(inputs)
                    outputs_scalar, mc_logits = split_model_outputs(outputs)
                    scalar_samples.append(outputs_scalar.unsqueeze(0))
                    if mc_logits is None:
                        continue
                    if isinstance(mc_logits, dict):
                        for task_idx, logits in mc_logits.items():
                            if logits is None:
                                continue
                            key = int(task_idx)
                            if key not in mc_logits_samples:
                                mc_logits_samples[key] = []
                            mc_logits_samples[key].append(logits.unsqueeze(0))
                    else:
                        num_tasks = mc_logits.size(1)
                        for j in range(num_tasks):
                            key = int(j)
                            if key not in mc_logits_samples:
                                mc_logits_samples[key] = []
                            mc_logits_samples[key].append(
                                mc_logits[:, j, :].unsqueeze(0)
                                )
                scalar_samples = torch.cat(scalar_samples, dim=0)
                _, batch_size, num_tasks = scalar_samples.shape
                smiles_all.extend(_batch_smiles(batch, batch_size))
                scalar_mean = scalar_samples.mean(dim=0)
                scalar_var = scalar_samples.var(dim=0, unbiased=False)
                scalar_std = torch.sqrt(scalar_var.clamp(min=0.0))
                tt = _resolve_task_type_for_batch(
                    model, task_type, y_true, scalar_samples[0]
                    )
                label_vals = mc_label_values
                if label_vals is None:
                    label_vals = getattr(model, "mc_label_values", None)
                batch_out = init_batch_uncertainty(
                    batch_size=batch_size,
                    num_tasks=num_tasks,
                    device=scalar_samples.device,
                    dtype=scalar_samples.dtype
                    )
                batch_out["pred_mean"] = scalar_mean
                batch_out["pred_var"] = scalar_var
                batch_out["pred_std"] = scalar_std

                is_bin = (tt == 1)
                is_mc = (tt == 2)
                is_reg = (tt == 0)

                if is_bin.any():
                    task_thresholds = _resolve_thresholds(
                        thresholds,
                        num_tasks,
                        device=scalar_samples.device,
                        dtype=scalar_samples.dtype,
                    )
                    p = torch.sigmoid(
                        scalar_samples[:, :, is_bin]
                        )
                    thr = task_thresholds[is_bin].view(1, -1)
                    p_mean = p.mean(dim=0)
                    p_var = p.var(dim=0, unbiased=False)
                    pred_entropy = _binary_entropy(p_mean)
                    expected_entropy = _binary_entropy(
                        p).mean(dim=0)
                    positive_rate = (p >= thr).to(p.dtype).mean(dim=0)
                    threshold_instability = (
                        1.0 - torch.abs(2.0 * positive_rate - 1.0)
                        )
                    decision_margin = p_mean - thr.squeeze(0)
                    denom = torch.where(
                        p_mean >= thr.squeeze(0),
                        1.0 - thr.squeeze(0),
                        thr.squeeze(0),
                    ).clamp_min(1e-6)
                    confidence_margin = (
                        torch.abs(decision_margin) / denom
                    ).clamp(max=1.0)
                    batch_out["pred_mean"][:, is_bin] = p_mean
                    batch_out["pred_var"][:, is_bin] = p_var
                    batch_out["pred_std"][:, is_bin] = torch.sqrt(
                        p_var.clamp(min=0.0)
                        )
                    batch_out["bin_var"][:, is_bin] = p_var
                    batch_out["bin_entropy"][:, is_bin] = pred_entropy
                    batch_out["bin_mutual_info"][:, is_bin] = torch.clamp(
                        pred_entropy - expected_entropy,
                        min=0.0
                        )
                    batch_out["bin_threshold"][:, is_bin] = (
                        thr.expand_as(p_mean)
                    )
                    batch_out["bin_decision_margin"][:, is_bin] = (
                        decision_margin
                    )
                    batch_out["bin_confidence_margin"][:, is_bin] = (
                        confidence_margin
                    )
                    batch_out["bin_mc_positive_rate"][:, is_bin] = (
                        positive_rate
                    )
                    batch_out["bin_threshold_instability"][:, is_bin] = (
                        threshold_instability
                    )

                if is_reg.any():
                    r = scalar_samples[:, :, is_reg]
                    r_mean = r.mean(dim=0)
                    r_var = r.var(dim=0, unbiased=False)
                    batch_out["pred_mean"][:, is_reg] = r_mean
                    batch_out["reg_var"][:, is_reg] = r_var
                    batch_out["reg_std"][:, is_reg] = torch.sqrt(
                        r_var.clamp(min=0.0)
                        )

                if is_mc.any():
                    for j in torch.where(is_mc)[0].tolist():
                        logits_list = mc_logits_samples.get(int(j), None)
                        if not logits_list:
                            continue
                        logits_j = torch.cat(logits_list, dim=0)
                        probs_j = torch.softmax(logits_j, dim=-1)
                        probs_mean = probs_j.mean(dim=0)
                        cls_idx = probs_mean.argmax(dim=-1)
                        if (
                            label_vals is not None
                            and j < len(label_vals)
                            and label_vals[j] is not None):
                            
                            values = torch.tensor(
                                label_vals[j],
                                device=cls_idx.device,
                                dtype=batch_out["pred_mean"].dtype
                                )
                            pred_j = values[cls_idx]
                        else:
                            pred_j = cls_idx.to(
                                batch_out["pred_mean"].dtype
                                )

                        pred_entropy = categorical_entropy(
                            probs_mean, dim=-1
                            )
                        expected_entropy = categorical_entropy(
                            probs_j, dim=-1
                            ).mean(dim=0)
                        pred_classes = probs_j.argmax(dim=-1)  # [S, B]
                        counts = F.one_hot(
                            pred_classes, num_classes=probs_j.size(-1)
                            ).sum(dim=0)
                        freq_max = counts.max(dim=-1).values.to(
                            probs_mean.dtype
                            )
                        variation_ratio = (
                            1.0 - freq_max / float(probs_j.size(0))
                            )

                        batch_out["pred_mean"][:, j] = pred_j
                        batch_out["mc_entropy"][:, j] = pred_entropy
                        batch_out["mc_expected_entropy"][:, j] = (
                            expected_entropy
                            )
                        batch_out["mc_mutual_info"][:, j] = torch.clamp(
                            pred_entropy - expected_entropy,
                            min=0.0
                            )
                        batch_out["mc_variation_ratio"][:, j] = (
                            variation_ratio
                            )

                        if j not in mc_prob_mean:
                            mc_prob_mean[j] = []
                        mc_prob_mean[j].append(
                            probs_mean.detach().cpu().numpy()
                            )

                pred_mean_all.append(
                    batch_out["pred_mean"].detach().cpu().numpy()
                    )
                pred_var_all.append(
                    batch_out["pred_var"].detach().cpu().numpy()
                    )
                pred_std_all.append(
                    batch_out["pred_std"].detach().cpu().numpy()
                    )
                bin_var_all.append(
                    batch_out["bin_var"].detach().cpu().numpy()
                    )
                bin_entropy_all.append(
                    batch_out["bin_entropy"].detach().cpu().numpy()
                    )
                bin_mi_all.append(
                    batch_out["bin_mutual_info"].detach().cpu().numpy()
                    )
                bin_threshold_all.append(
                    batch_out["bin_threshold"].detach().cpu().numpy()
                    )
                bin_decision_margin_all.append(
                    batch_out[
                        "bin_decision_margin"].detach().cpu().numpy()
                    )
                bin_confidence_margin_all.append(
                    batch_out[
                        "bin_confidence_margin"].detach().cpu().numpy()
                    )
                bin_mc_positive_rate_all.append(
                    batch_out[
                        "bin_mc_positive_rate"].detach().cpu().numpy()
                    )
                bin_threshold_instability_all.append(
                    batch_out[
                        "bin_threshold_instability"].detach().cpu().numpy()
                    )
                reg_var_all.append(
                    batch_out["reg_var"].detach().cpu().numpy()
                    )
                reg_std_all.append(
                    batch_out["reg_std"].detach().cpu().numpy()
                    )
                mc_entropy_all.append(
                    batch_out["mc_entropy"].detach().cpu().numpy()
                    )
                mc_expected_entropy_all.append(
                    batch_out["mc_expected_entropy"].detach().cpu().numpy()
                    )
                mc_mi_all.append(
                    batch_out["mc_mutual_info"].detach().cpu().numpy()
                    )
                mc_vr_all.append(
                    batch_out["mc_variation_ratio"].detach().cpu().numpy()
                    )
    finally:
        restore_dropout(model, prev_mode, dropout_layers)

    pred_mean = _stack_or_none(pred_mean_all)
    payload = {
        "pred_mean": pred_mean,
        "pred_var": _stack_or_none(pred_var_all),
        "pred_std": _stack_or_none(pred_std_all),
        "labels": _stack_or_none(labels),
        "bin_var": _stack_or_none(bin_var_all),
        "bin_entropy": _stack_or_none(bin_entropy_all),
        "bin_mutual_info": _stack_or_none(bin_mi_all),
        "bin_threshold": _stack_or_none(
            bin_threshold_all),
        "bin_decision_margin": _stack_or_none(
            bin_decision_margin_all),
        "bin_confidence_margin": _stack_or_none(
            bin_confidence_margin_all),
        "bin_mc_positive_rate": _stack_or_none(
            bin_mc_positive_rate_all),
        "bin_threshold_instability": _stack_or_none(
            bin_threshold_instability_all),
        "reg_var": _stack_or_none(reg_var_all),
        "reg_std": _stack_or_none(reg_std_all),
        "mc_entropy": _stack_or_none(mc_entropy_all),
        "mc_expected_entropy": _stack_or_none(mc_expected_entropy_all),
        "mc_mutual_info": _stack_or_none(mc_mi_all),
        "mc_variation_ratio": _stack_or_none(mc_vr_all),
        "mc_prob_mean": {k: np.concatenate(
            v, axis=0) for k, v in mc_prob_mean.items()},
        "smiles": (smiles_all if (pred_mean is not None
                and smiles_all and len(smiles_all) == pred_mean.shape[0])
            else None),
        "task_aleatoric": None,
        "task_log_vars": None}
    uncertainty_module = getattr(model, "uncertainty", None)
    if uncertainty_module is not None and hasattr(
        uncertainty_module, "log_vars"):

        log_vars = uncertainty_module.log_vars.detach().cpu().numpy()
        payload["task_log_vars"] = log_vars
        payload["task_aleatoric"] = np.exp(log_vars)
        
    return payload


def predict_with_uncertainty(
    model,
    data_loader,
    device,
    n_samples=30,
    task_type=None,
    mc_label_values=None,
    mc_labels=None,
    n_mc_samples=None,
    use_thresholds=False,
    thresholds_path="../output/calibration/thresholds.json",
    thresholds_split="val",
    thresholds=None):
    return predict_uncertainty(
        model,
        data_loader,
        device,
        n_samples=n_samples,
        task_type=task_type,
        mc_label_values=(mc_label_values if mc_label_values is not None else mc_labels),
        n_mc_samples=n_mc_samples,
        use_thresholds=use_thresholds,
        thresholds_path=thresholds_path,
        thresholds_split=thresholds_split,
        thresholds=thresholds,
    )


def col_or_nan(arr, task_idx, n_rows):
    if arr is None:
        return np.full(int(n_rows), np.nan, dtype=float)
    mat = np.asarray(arr)
    if mat.ndim != 2 or int(task_idx) >= mat.shape[1]:
        return np.full(int(n_rows), np.nan, dtype=float)
    return mat[:, int(task_idx)]


def _task_kind(task_type, task_idx):
    if task_type is None:
        return None
    if not torch.is_tensor(task_type):
        task_type = torch.tensor(task_type, dtype=torch.long)
    idx = int(task_idx)
    if idx < 0 or idx >= int(task_type.numel()):
        return None
    val = int(task_type[idx].item())
    if val == 2:
        return "multiclass"
    if val == 1:
        return "binary"
    return "regression"


def _save_figure(fig, out_path):
    if out_path is None:
        return None
    out_path = str(out_path)
    folder = os.path.dirname(out_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.savefig(out_path, dpi=600)
    return out_path


def _hist_heights(values, bins, hist_range, density=False):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    counts, edges = np.histogram(vals, bins=bins, range=hist_range)
    counts = counts.astype(float)
    if density and vals.size > 0:
        widths = np.diff(edges)
        denom = counts.sum() * widths
        counts = np.divide(
            counts,
            denom,
            out=np.zeros_like(counts),
            where=denom > 0,
        )
    return counts, edges


def _plot_overlapped_hist(
    ax,
    hist_items,
    bins,
    hist_range,
    density=False,
    orientation="vertical",
    alpha=0.55,
    linewidth=0.4):

    computed = []
    edges = None
    for item in hist_items:
        values = item["values"]
        heights, edges = _hist_heights(
            values,
            bins=bins,
            hist_range=hist_range,
            density=density,
        )
        computed.append((heights, item))
    if edges is None:
        return
    widths = np.diff(edges)
    for bin_idx in range(len(edges) - 1):
        ordered = sorted(
            computed,
            key=lambda pair: pair[0][bin_idx],
            reverse=True,
        )
        for z_idx, (heights, item) in enumerate(ordered):
            height = heights[bin_idx]
            if height <= 0:
                continue
            if orientation == "horizontal":
                ax.barh(
                    edges[bin_idx],
                    height,
                    height=widths[bin_idx],
                    align="edge",
                    color=item["color"],
                    edgecolor=item.get("edgecolor", item["color"]),
                    alpha=alpha,
                    linewidth=linewidth,
                    zorder=2 + z_idx,
                )
            else:
                ax.bar(
                    edges[bin_idx],
                    height,
                    width=widths[bin_idx],
                    align="edge",
                    color=item["color"],
                    edgecolor=item.get("edgecolor", item["color"]),
                    alpha=alpha,
                    linewidth=linewidth,
                    zorder=2 + z_idx,
                )


def _binary_task_indices(task_type, n_tasks):
    if task_type is None:
        return list(range(int(n_tasks)))
    if not torch.is_tensor(task_type):
        task_type = torch.tensor(task_type, dtype=torch.long)
    return [
        int(i) for i in torch.where(task_type.to(torch.long) == 1)[0]
        if int(i) < int(n_tasks)
    ]


def plot_binary_entropy_histograms(
    uncertainty,
    out_path,
    task_type=None,
    task_indices=None,
    max_tasks=9,
    bins=25):

    entropy = uncertainty.get("bin_entropy", None)
    labels = uncertainty.get("labels", None)
    pred_mean = uncertainty.get("pred_mean", None)
    thresholds = uncertainty.get("bin_threshold", None)
    if entropy is None:
        raise ValueError("Missing binary entropy values.")
    entropy = np.asarray(entropy, dtype=float) / np.log(2.0)
    n_tasks = entropy.shape[1]
    if task_indices is None:
        task_indices = _binary_task_indices(task_type, n_tasks)
    task_indices = [int(t) for t in task_indices[:max_tasks]]
    if not task_indices:
        raise ValueError("No binary tasks available for entropy plots.")

    labels = None if labels is None else np.asarray(labels, dtype=float)
    pred_mean = None if pred_mean is None else np.asarray(pred_mean, dtype=float)
    thresholds = (
        None if thresholds is None else np.asarray(thresholds, dtype=float)
    )
    ncols = min(3, len(task_indices))
    nrows = int(np.ceil(len(task_indices) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.0, 3.2 * nrows),
        squeeze=False,
    )
    colors = {"inactive": _INACTIVE_COLOR, "active": _ACTIVE_COLOR}
    for ax, task_idx in zip(axes.ravel(), task_indices):
        vals = entropy[:, task_idx]
        valid = np.isfinite(vals)
        if labels is not None and labels.ndim == 2:
            y = labels[:, task_idx]
            labeled = valid & np.isfinite(y)
            inactive = labeled & (y < 0.5)
            active = labeled & (y >= 0.5)
            unlabeled = valid & ~np.isfinite(y)
        else:
            unlabeled = np.zeros_like(valid, dtype=bool)
            if pred_mean is None:
                inactive = valid
                active = np.zeros_like(valid, dtype=bool)
            else:
                thr = 0.5
                if thresholds is not None:
                    tvals = thresholds[:, task_idx]
                    if np.isfinite(tvals).any():
                        thr = float(np.nanmedian(tvals))
                inactive = valid & (pred_mean[:, task_idx] < thr)
                active = valid & (pred_mean[:, task_idx] >= thr)
        hist_items = []
        if unlabeled.any():
            hist_items.append({
                "values": vals[unlabeled],
                "color": _UNLABELED_COLOR,
                "edgecolor": "#555555",
                "label": "Unlabeled",
            })
        hist_items.extend([
            {
                "values": vals[inactive],
                "color": colors["inactive"],
                "edgecolor": colors["inactive"],
                "label": "Inactive",
            },
            {
                "values": vals[active],
                "color": colors["active"],
                "edgecolor": colors["active"],
                "label": "Active",
            },
        ])
        _plot_overlapped_hist(
            ax,
            hist_items,
            bins=bins,
            hist_range=(0.0, 1.0),
            density=True,
            orientation="vertical",
            alpha=0.48,
            linewidth=0.4,
        )
        ax.set_xlim(0.0, 1.0)
        ax.tick_params(labelsize=18)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        if task_idx == task_indices[0]:
            legend_handles = [
                Patch(
                    facecolor=colors["inactive"],
                    edgecolor=colors["inactive"],
                    alpha=0.48,
                    label="Inactive",
                ),
                Patch(
                    facecolor=colors["active"],
                    edgecolor=colors["active"],
                    alpha=0.48,
                    label="Active",
                ),
            ]
            if unlabeled.any():
                legend_handles.append(
                    Patch(
                        facecolor=_UNLABELED_COLOR,
                        edgecolor="#555555",
                        alpha=0.48,
                        label="Unlabeled",
                    )
                )
            leg = ax.legend(
                handles=legend_handles,
                loc="upper right",
                frameon=True,
                fontsize=10,
            )
            fr = leg.get_frame()
            fr.set_facecolor("white")
            fr.set_edgecolor("black")
            fr.set_linewidth(1.0)
    for ax in axes.ravel()[len(task_indices):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("Normalized Entropy", fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel("Density", fontsize=12)
    fig.tight_layout()
    _save_figure(fig, out_path)
    return fig


def plot_binary_threshold_scatter(
    uncertainty,
    out_path,
    task_idx=0,
    uncertainty_cutoff=0.5,
    bins=28):

    pred = col_or_nan(
        uncertainty.get("pred_mean"),
        task_idx,
        np.asarray(uncertainty["pred_mean"]).shape[0],
    )
    mi = col_or_nan(
        uncertainty.get("bin_mutual_info"), task_idx, pred.shape[0])
    margin = col_or_nan(
        uncertainty.get("bin_confidence_margin"), task_idx, pred.shape[0])
    instability = col_or_nan(
        uncertainty.get("bin_threshold_instability"), task_idx, pred.shape[0])
    thresholds = col_or_nan(
        uncertainty.get("bin_threshold"), task_idx, pred.shape[0])
    labels = uncertainty.get("labels", None)
    if labels is not None:
        labels = np.asarray(labels, dtype=float)
        y = labels[:, int(task_idx)]
    else:
        y = np.full(pred.shape[0], np.nan, dtype=float)
    valid = (
        np.isfinite(pred)
        & np.isfinite(mi)
        & np.isfinite(margin)
        & np.isfinite(instability)
        & np.isfinite(thresholds)
    )
    if not valid.any():
        raise ValueError("No valid binary uncertainty values to plot.")
    print(
        f"Threshold scatter task {int(task_idx)}: "
        f"plotting {int(valid.sum())}/{int(valid.size)} samples."
    )
    x_vals = mi[valid]
    y_vals = margin[valid]
    xmin, xmax = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
    ymin, ymax = float(np.nanmin(y_vals)), float(np.nanmax(y_vals))
    x_span = max(xmax - xmin, 1e-9)
    y_span = max(ymax - ymin, 1e-9)
    x_pad = max(0.04 * x_span, 1e-4)
    y_pad = max(0.04 * y_span, 1e-4)
    xmin = xmin - x_pad
    xmax = xmax + x_pad
    ymin = ymin - y_pad
    ymax = ymax + y_pad
    pred_label = pred >= thresholds
    has_label = np.isfinite(y)
    true_label = y >= 0.5
    cases = {
        "True negative": valid & has_label & ~true_label & ~pred_label,
        "False positive": valid & has_label & ~true_label & pred_label,
        "False negative": valid & has_label & true_label & ~pred_label,
        "True positive": valid & has_label & true_label & pred_label,
        "Unlabeled": valid & ~has_label,
    }
    legend_order = [
        "True positive",
        "True negative",
        "False positive",
        "False negative",
    ]
    colors = {
        "True negative": _INACTIVE_COLOR,
        "False positive": _ACTIVE_COLOR,
        "False negative": _INACTIVE_COLOR,
        "True positive": _ACTIVE_COLOR,
        "Unlabeled": _UNLABELED_COLOR,
    }
    edgecolors = {
        "True negative": "none",
        "False positive": "black",
        "False negative": "black",
        "True positive": "none",
        "Unlabeled": "none",
    }

    n_valid = int(valid.sum())
    n_total = int(valid.size)
    fig = plt.figure(figsize=(7.0, 6.2))
    gs = GridSpec(
        2,
        2,
        width_ratios=(4.0, 1.0),
        height_ratios=(1.0, 4.0),
        hspace=0.05,
        wspace=0.05,
    )
    ax_histx = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_histx)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    for name, mask in cases.items():
        if not mask.any():
            continue
        ax.scatter(
            mi[mask],
            margin[mask],
            s=28.0 + 120.0 * instability[mask],
            marker="o",
            color=colors[name],
            alpha=0.72,
            label=name,
            edgecolor=edgecolors[name],
            linewidth=1.15 if edgecolors[name] != "none" else 0.0,
        )
    hist_items_x = [
        {
            "values": mi[mask],
            "color": colors[name],
            "edgecolor": (
                "black" if edgecolors[name] != "none" else colors[name]
            ),
            "label": name,
        }
        for name, mask in cases.items()
        if mask.any()
    ]
    hist_items_y = [
        {
            "values": margin[mask],
            "color": colors[name],
            "edgecolor": (
                "black" if edgecolors[name] != "none" else colors[name]
            ),
            "label": name,
        }
        for name, mask in cases.items()
        if mask.any()
    ]
    _plot_overlapped_hist(
        ax_histx,
        hist_items_x,
        bins=bins,
        hist_range=(xmin, xmax),
        density=False,
        orientation="vertical",
        alpha=0.65,
        linewidth=0.55,
    )
    _plot_overlapped_hist(
        ax_histy,
        hist_items_y,
        bins=bins,
        hist_range=(ymin, ymax),
        density=False,
        orientation="horizontal",
        alpha=0.65,
        linewidth=0.55,
    )
    ax.set_xlabel(
        f"Mutual Information (mean={np.nanmean(mi[valid]):.2f})",
        fontsize=12,
    )
    ax.set_ylabel(
        f"Margin of confidence (mean={np.nanmean(margin[valid]):.2f})",
        fontsize=12,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            label=name,
            markerfacecolor=colors[name],
            markeredgecolor=edgecolors[name],
            markeredgewidth=1.15 if edgecolors[name] != "none" else 0.0,
            markersize=6.5,
            alpha=0.72,
        )
        for name in legend_order
    ]
    if cases.get("Unlabeled", np.zeros_like(valid)).any():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                label="Unlabeled",
                markerfacecolor=_UNLABELED_COLOR,
                markeredgecolor="none",
                markersize=6.5,
                alpha=0.72,
            )
        )
    leg = ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=11,
        frameon=True,
        markerscale=0.9,
        handletextpad=0.4,
        borderpad=0.4,
    )
    fr = leg.get_frame()
    fr.set_facecolor("white")
    fr.set_edgecolor("black")
    fr.set_linewidth(1.0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax_histx.set_xlim(xmin, xmax)
    ax_histy.set_ylim(ymin, ymax)
    ax.tick_params(labelsize=18)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_histx.tick_params(
        axis="x", labelbottom=False, labelsize=10)
    ax_histx.tick_params(
        axis="y", labelleft=True, labelsize=10)
    ax_histy.tick_params(
        axis="y", labelleft=False, labelsize=10)
    ax_histy.tick_params(
        axis="x", labelbottom=True, labelsize=10)
    ax_histx.yaxis.set_major_locator(MaxNLocator(nbins=3))
    x0, x1 = ax_histy.get_xlim()
    if np.isfinite(x0) and np.isfinite(x1) and x1 > x0:
        ax_histy.xaxis.set_major_locator(
            FixedLocator(np.linspace(x0, x1, 4)[1:-1])
        )
    else:
        ax_histy.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_histx.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax_histy.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    for marginal_ax in (ax_histx, ax_histy):
        for spine in marginal_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
    fig.tight_layout()
    _save_figure(fig, out_path)
    return fig


def _concat_uncertainty_payloads(payloads):
    payloads = [p for p in payloads if p is not None]
    if not payloads:
        raise ValueError("No uncertainty payloads to concatenate.")

    keys = set()
    for payload in payloads:
        keys.update(payload.keys())

    combined = {}
    for key in keys:
        values = [payload.get(key, None) for payload in payloads]
        valid_values = [value for value in values if value is not None]
        if not valid_values:
            combined[key] = None
            continue
        if key == "mc_prob_mean":
            merged = {}
            all_mc_keys = set()
            for value in valid_values:
                all_mc_keys.update(value.keys())
            for mc_key in all_mc_keys:
                arrs = [
                    value[mc_key] for value in valid_values
                    if mc_key in value
                ]
                merged[mc_key] = np.concatenate(arrs, axis=0)
            combined[key] = merged
        elif key == "smiles":
            smiles = []
            for value in values:
                if value is not None:
                    smiles.extend(value)
            combined[key] = smiles if smiles else None
        elif key in ("task_aleatoric", "task_log_vars"):
            combined[key] = valid_values[0]
        else:
            try:
                combined[key] = np.concatenate(valid_values, axis=0)
            except Exception:
                combined[key] = valid_values[0]
    return combined


def view_uncertainty_splits(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    plots_dir,
    task_idx=0,
    n_samples=30,
    task_type=None,
    mc_label_values=None,
    mc_labels=None,
    n_mc_samples=None,
    use_thresholds=False,
    thresholds_path="../output/calibration/thresholds.json",
    thresholds_split="val",
    thresholds=None,
    uncertainty_cutoff=0.5,
    out_path=None):

    if n_mc_samples is not None:
        n_samples = n_mc_samples
    if mc_label_values is None and mc_labels is not None:
        mc_label_values = mc_labels
    if task_type is None:
        task_type = getattr(model, "task_type", None)
    if mc_label_values is None:
        mc_label_values = getattr(model, "mc_label_values", None)
    if task_type is None or mc_label_values is None:
        try:
            from loaders import infer_task_metadata
        except Exception:
            from train.loaders import infer_task_metadata
        inferred_type, _, inferred_labels = infer_task_metadata(train_loader)
        if task_type is None and inferred_type is not None:
            task_type = inferred_type
        if mc_label_values is None and inferred_labels is not None:
            mc_label_values = inferred_labels

    split_loaders = (
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    )
    payloads = []
    split_names = []
    for split_name, loader in split_loaders:
        if loader is None:
            continue
        payload = predict_uncertainty(
            model,
            loader,
            device,
            n_samples=n_samples,
            task_type=task_type,
            mc_label_values=mc_label_values,
            use_thresholds=use_thresholds,
            thresholds_path=thresholds_path,
            thresholds_split=thresholds_split,
            thresholds=thresholds,
        )
        pred_mean = payload.get("pred_mean", None)
        if pred_mean is not None:
            split_names.extend([split_name] * int(pred_mean.shape[0]))
        payloads.append(payload)

    uncertainty = _concat_uncertainty_payloads(payloads)
    if split_names:
        uncertainty["split"] = np.asarray(split_names)

    pred_mean = uncertainty.get("pred_mean", None)
    if out_path is not None:
        if pred_mean is None:
            raise ValueError("No predictions were generated.")
        task_idx = int(task_idx)
        n_rows = int(pred_mean.shape[0])
        task_suffix = f"t{task_idx}"
        data = {
            "sample_idx": np.arange(n_rows, dtype=int),
            "split": uncertainty.get(
                "split",
                np.full(n_rows, None, dtype=object),
            ),
            f"y_pred_mean_{task_suffix}": col_or_nan(
                uncertainty.get("pred_mean"), task_idx, n_rows),
            f"pred_std_{task_suffix}": col_or_nan(
                uncertainty.get("pred_std"), task_idx, n_rows),
            f"pred_var_{task_suffix}": col_or_nan(
                uncertainty.get("pred_var"), task_idx, n_rows),
            f"bin_entropy_{task_suffix}": col_or_nan(
                uncertainty.get("bin_entropy"), task_idx, n_rows),
            f"bin_mutual_info_{task_suffix}": col_or_nan(
                uncertainty.get("bin_mutual_info"), task_idx, n_rows),
            f"bin_var_{task_suffix}": col_or_nan(
                uncertainty.get("bin_var"), task_idx, n_rows),
            f"threshold_{task_suffix}": col_or_nan(
                uncertainty.get("bin_threshold"), task_idx, n_rows),
            f"decision_margin_{task_suffix}": col_or_nan(
                uncertainty.get("bin_decision_margin"), task_idx, n_rows),
            f"confidence_margin_{task_suffix}": col_or_nan(
                uncertainty.get("bin_confidence_margin"), task_idx, n_rows),
            f"mc_positive_rate_{task_suffix}": col_or_nan(
                uncertainty.get("bin_mc_positive_rate"), task_idx, n_rows),
            f"threshold_instability_{task_suffix}": col_or_nan(
                uncertainty.get("bin_threshold_instability"),
                task_idx,
                n_rows,
            ),
        }
        labels = uncertainty.get("labels", None)
        if labels is not None:
            labels = np.asarray(labels)
            if labels.ndim == 2 and task_idx < labels.shape[1]:
                data[f"y_true_{task_suffix}"] = labels[:, task_idx]
        smiles = uncertainty.get("smiles", None)
        if smiles is not None and len(smiles) == n_rows:
            data["smiles"] = smiles
        y_pred_task = data[f"y_pred_mean_{task_suffix}"]
        threshold_task = data[f"threshold_{task_suffix}"]
        data[f"y_pred_label_{task_suffix}"] = np.where(
            np.isnan(y_pred_task),
            np.nan,
            (y_pred_task >= threshold_task).astype(float),
        )
        df = pd.DataFrame(data)
        out_path = str(out_path)
        if not out_path.lower().endswith(".xlsx"):
            raise ValueError("out_path must end with .xlsx")
        folder = os.path.dirname(out_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        df.to_excel(out_path, index=False)
        print(f"Uncertainty table saved to {out_path}")

    os.makedirs(plots_dir, exist_ok=True)
    task_suffix = f"t{int(task_idx)}"
    plot_binary_threshold_scatter(
        uncertainty,
        os.path.join(
            plots_dir,
            f"uncertainty_threshold_scatter_all_splits_{task_suffix}.svg",
        ),
        task_idx=task_idx,
        uncertainty_cutoff=uncertainty_cutoff,
    )
    plot_binary_entropy_histograms(
        uncertainty,
        os.path.join(
            plots_dir,
            f"uncertainty_entropy_histogram_all_splits_{task_suffix}.svg",
        ),
        task_type=task_type,
        task_indices=[task_idx],
    )
    return uncertainty


def view_uncertainty(
    model,
    loader,
    device,
    out_path,
    task_idx=0,
    n_samples=30,
    task_type=None,
    mc_label_values=None,
    mc_labels=None,
    n_mc_samples=None,
    use_thresholds=False,
    thresholds_path="../output/calibration/thresholds.json",
    thresholds_split="val",
    thresholds=None,
    make_plots=False,
    plots_dir=None,
    uncertainty_cutoff=0.5):
    if n_mc_samples is not None:
        n_samples = n_mc_samples
    if mc_label_values is None and mc_labels is not None:
        mc_label_values = mc_labels
    model = model.to(device)
    if task_type is None:
        task_type = getattr(model, "task_type", None)
    if mc_label_values is None:
        mc_label_values = getattr(
            model, "mc_label_values", None)
    if task_type is None or mc_label_values is None:
        try:
            from loaders import infer_task_metadata
        except Exception:
            from train.loaders import infer_task_metadata

        inferred_type, _, inferred_labels = infer_task_metadata(loader)
        if task_type is None and inferred_type is not None:
            task_type = inferred_type
        if mc_label_values is None and inferred_labels is not None:
            mc_label_values = inferred_labels
    if task_type is not None:
        model.task_type = task_type
    if mc_label_values is not None:
        model.mc_label_values = mc_label_values
    has_mc_head = True
    task_kind = _task_kind(task_type, task_idx)
    if task_kind == "multiclass":
        mc_heads = getattr(model, "mc_heads", None)
        if mc_heads is None or str(int(task_idx)) not in mc_heads:
            has_mc_head = False

    uncertainty = predict_uncertainty(
        model,
        loader,
        device,
        n_samples=n_samples,
        task_type=task_type,
        mc_label_values=mc_label_values,
        use_thresholds=use_thresholds,
        thresholds_path=thresholds_path,
        thresholds_split=thresholds_split,
        thresholds=thresholds,
        )
    pred_mean = uncertainty.get("pred_mean", None)
    if pred_mean is None:
        raise ValueError("No predictions were generated.")
    task_idx = int(task_idx)
    if task_idx < 0 or task_idx >= pred_mean.shape[1]:
        raise ValueError(
            f"task_idx={task_idx} out of range "
            f"[0, {pred_mean.shape[1] - 1}]"
            )
    n_rows = pred_mean.shape[0]
    task_suffix = f"t{task_idx}"
    y_pred_task = col_or_nan(
        uncertainty.get("pred_mean"), task_idx, n_rows)
    y_pred_export = y_pred_task
    if task_kind == "multiclass" and not has_mc_head:
        y_pred_export = np.full(int(n_rows), np.nan, dtype=float)
    pred_std_task = col_or_nan(
        uncertainty.get("pred_std"), task_idx, n_rows)
    pred_var_task = col_or_nan(
        uncertainty.get("pred_var"), task_idx, n_rows)
    data = {
        "sample_idx": np.arange(n_rows, dtype=int),
        f"y_pred_mean_{task_suffix}": y_pred_export,
        f"pred_std_{task_suffix}": pred_std_task,
        f"pred_var_{task_suffix}": pred_var_task,
        }
    labels = uncertainty.get("labels", None)
    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim == 2 and task_idx < labels.shape[1]:
            data[f"y_true_{task_suffix}"] = labels[:, task_idx]
    smiles = uncertainty.get("smiles", None)
    if smiles is not None and len(smiles) == n_rows:
        data["smiles"] = smiles

    if task_kind == "binary":
        bin_entropy = col_or_nan(
            uncertainty.get("bin_entropy"), task_idx, n_rows)
        bin_mi = col_or_nan(
            uncertainty.get("bin_mutual_info"), task_idx, n_rows)
        bin_var = col_or_nan(
            uncertainty.get("bin_var"), task_idx, n_rows)
        bin_threshold = col_or_nan(
            uncertainty.get("bin_threshold"), task_idx, n_rows)
        bin_decision_margin = col_or_nan(
            uncertainty.get("bin_decision_margin"), task_idx, n_rows)
        bin_confidence_margin = col_or_nan(
            uncertainty.get("bin_confidence_margin"), task_idx, n_rows)
        bin_mc_positive_rate = col_or_nan(
            uncertainty.get("bin_mc_positive_rate"), task_idx, n_rows)
        bin_threshold_instability = col_or_nan(
            uncertainty.get("bin_threshold_instability"), task_idx, n_rows)
        data[f"bin_entropy_{task_suffix}"] = bin_entropy
        data[f"bin_mutual_info_{task_suffix}"] = bin_mi
        data[f"bin_var_{task_suffix}"] = bin_var
        data[f"threshold_{task_suffix}"] = bin_threshold
        data[f"decision_margin_{task_suffix}"] = bin_decision_margin
        data[f"confidence_margin_{task_suffix}"] = bin_confidence_margin
        data[f"mc_positive_rate_{task_suffix}"] = bin_mc_positive_rate
        data[f"threshold_instability_{task_suffix}"] = (
            bin_threshold_instability
        )
        data[f"y_pred_label_{task_suffix}"] = np.where(
            np.isnan(y_pred_task), np.nan, (
                y_pred_task >= bin_threshold).astype(float))
        data[f"confidence_{task_suffix}"] = np.where(
            np.isnan(y_pred_task), np.nan, np.abs(
                y_pred_task - bin_threshold) * 2.0)
    elif task_kind == "multiclass" and has_mc_head:
        mc_entropy = col_or_nan(uncertainty.get(
            "mc_entropy"), task_idx, n_rows)
        mc_expected_entropy = col_or_nan(uncertainty.get(
            "mc_expected_entropy"), task_idx, n_rows)
        mc_mi = col_or_nan(uncertainty.get(
            "mc_mutual_info"), task_idx, n_rows)
        mc_vr = col_or_nan(uncertainty.get(
            "mc_variation_ratio"), task_idx, n_rows)
        data[f"mc_entropy_{task_suffix}"] = mc_entropy
        data[f"mc_expected_entropy_{task_suffix}"] = mc_expected_entropy
        data[f"mc_mutual_info_{task_suffix}"] = mc_mi
        data[f"mc_variation_ratio_{task_suffix}"] = mc_vr
        mc_prob_mean = uncertainty.get("mc_prob_mean", {})
        probs = mc_prob_mean.get(int(task_idx), None)
        if probs is None:
            probs = mc_prob_mean.get(str(int(task_idx)), None)
        if probs is not None:
            probs = np.asarray(probs)
            top_idx = probs.argmax(axis=1)
            top_prob = probs.max(axis=1)
            if (mc_label_values is not None
                and int(task_idx) < len(mc_label_values)
                and mc_label_values[int(task_idx)] is not None):
                values = np.asarray(mc_label_values[int(task_idx)])
                data[f"y_pred_class_{task_suffix}"] = values[top_idx]
            else:
                data[f"y_pred_class_{task_suffix}"] = top_idx
            data[f"y_pred_top1_prob_{task_suffix}"] = top_prob
    elif task_kind == "regression":
        reg_std = col_or_nan(
            uncertainty.get("reg_std"), task_idx, n_rows)
        reg_var = col_or_nan(
            uncertainty.get("reg_var"), task_idx, n_rows)
        data[f"reg_std_{task_suffix}"] = reg_std
        data[f"reg_var_{task_suffix}"] = reg_var
        std_used = np.where(np.isnan(reg_std), pred_std_task, reg_std)
        data[f"pi95_low_{task_suffix}"] = y_pred_task - 1.96 * std_used
        data[f"pi95_high_{task_suffix}"] = y_pred_task + 1.96 * std_used
    elif task_kind == "multiclass" and not has_mc_head:
        data[f"y_pred_scalar_mean_{task_suffix}"] = y_pred_task
        data[f"uncertainty_mode_{task_suffix}"] = "fallback_scalar"

    df = pd.DataFrame(data)

    out_path = str(out_path)
    if not out_path.lower().endswith(".xlsx"):
        raise ValueError("out_path must end with .xlsx")
    folder = os.path.dirname(out_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    if task_kind == "binary":
        cols = [
            f"bin_entropy_{task_suffix}",
            f"bin_mutual_info_{task_suffix}",
            f"bin_var_{task_suffix}",
            ]
    elif task_kind == "multiclass" and has_mc_head:
        cols = [
            f"mc_entropy_{task_suffix}",
            f"mc_mutual_info_{task_suffix}",
            f"mc_variation_ratio_{task_suffix}",
            ]
    elif task_kind == "regression":
        cols = [
            f"reg_std_{task_suffix}",
            f"reg_var_{task_suffix}",
            ]
    else:
        cols = []
    generic_cols = [
        f"pred_std_{task_suffix}",
        f"pred_var_{task_suffix}",
        ]
    if cols and all(df[c].notna().sum() == 0 for c in cols):
        if all(df[c].notna().sum() == 0 for c in generic_cols):
            raise ValueError("No uncertainty values were computed.")

    df.to_excel(out_path, index=False)
    print(f"Uncertainty table saved to {out_path}")
    if make_plots:
        if plots_dir is None:
            plots_dir = os.path.splitext(out_path)[0] + "_figures"
        os.makedirs(plots_dir, exist_ok=True)
        if task_kind == "binary":
            plot_binary_threshold_scatter(
                uncertainty,
                os.path.join(
                    plots_dir,
                    f"uncertainty_threshold_scatter_{task_suffix}.svg",
                ),
                task_idx=task_idx,
                uncertainty_cutoff=uncertainty_cutoff,
            )
            plot_binary_entropy_histograms(
                uncertainty,
                os.path.join(
                    plots_dir,
                    f"uncertainty_entropy_histogram_{task_suffix}.svg",
                ),
                task_type=task_type,
                task_indices=[task_idx],
            )
    return df
