import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch

from utils import device
from predictor import predict

from figures import (
    FigureConfig,
    SaveConfig,
    RcParamsConfig,
    Palette,
    to_2d,
    fix_orientation,
    style_spines,
    legend_frame,
    infer_task_kind,
    gather_labels
    )


def pred_from_loader(
    model, 
    loader, 
    device, 
    task_index, 
    inverse_true, 
    inverse_pred):

    y_pred, y_true, _ = predict(model, loader, device)
    y_true = fix_orientation(to_2d(y_true))
    y_pred = fix_orientation(to_2d(y_pred))
    if inverse_true is not None:
        y_true = inverse_true(y_true)
    if inverse_pred is not None:
        y_pred = inverse_pred(y_pred)
    x = y_true[:, task_index].astype(float)
    y = y_pred[:, task_index].astype(float)
    m = np.isfinite(x) & np.isfinite(y)

    return x[m], y[m]


def histogram_plot(
    model,
    train_loader,
    val_loader,
    test_loader,
    task_index=0,
    out_path=None,
    bins=25,
    fig_scale=None,
    dpi=None,
    palette=Palette(),
    rc=RcParamsConfig()): 
    
    rc.apply()
    fig_cfg = FigureConfig(fig_scale=fig_scale)
    save_cfg = SaveConfig(
        out_path=out_path, dpi=dpi, suffix="_prob"
        )
    range = (0.0, 1.0)
    x_tr, y_pred_tr = pred_from_loader(
        model, train_loader, device,
        task_index, None, None
        )
    x_va, y_pred_va = pred_from_loader(
        model, val_loader, device,
        task_index, None, None
        )
    x_te, y_pred_te = pred_from_loader(
        model, test_loader, device,
        task_index, None, None
        )
    y_prob_tr = np.clip(y_pred_tr, 0.0, 1.0)
    y_prob_va = np.clip(y_pred_va, 0.0, 1.0)
    y_prob_te = np.clip(y_pred_te, 0.0, 1.0)
    y_true_tr = x_tr.astype(bool)
    y_true_va = x_va.astype(bool)
    y_true_te = x_te.astype(bool)

    sets = [
        ("Train", y_prob_tr, y_true_tr),
        ("Validation", y_prob_va, y_true_va),
        ("Test", y_prob_te, y_true_te),
        ]
    fig = plt.figure(
        figsize=(fig_cfg.side(), fig_cfg.side())
        )
    gs = gridspec.GridSpec(
        nrows=3, ncols=1,
        height_ratios=[1, 1, 1],
        hspace=0.12
        )
    for i, (name, p, y) in enumerate(sets):
        ax = fig.add_subplot(gs[i, 0])
        if p.size == 0:
            ax.text(0.5, 0.5, f"{name}: vazio",
                ha="center", va="center"
                )
            ax.set_axis_off()
            continue
        neg = p[~y]
        pos = p[y]
        h1 = ax.hist(neg, bins=bins, range=range,
            alpha=0.85, color=palette.inact
            )
        h2 = ax.hist(pos, bins=bins, range=range,
            alpha=0.55, color=palette.act
            )
        ax.set_xlim(range)
        ym = 0
        if len(h1[0]):
            ym = max(ym, int(np.ceil(h1[0].max())))
        if len(h2[0]):
            ym = max(ym, int(np.ceil(h2[0].max())))
        ym = max(ym, 1)
        ax.set_ylim(0, ym)
        ax.set_yticks([0, ym])
        ax.set_xticks([0, 0.5, 1.0])
        if i < 2:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Count")
        style_spines(ax)
        ax.grid(False)
        hdl = [
            Patch(facecolor=palette.inact,
                alpha=0.85, label="Inactive"),
            Patch(facecolor=palette.act,
                alpha=0.55, label="Active"),
            ]
        leg = ax.legend(handles=hdl, loc="upper left", 
            frameon=True, fontsize=rc.legend_fontsize
            )
        fr = leg.get_frame()
        fr.set_facecolor("white")
        fr.set_edgecolor("black")
        fr.set_linewidth(1.0)
    plt.tight_layout()
    save_cfg.maybe_save(fig)
    plt.show()


def scatter_plot(
    model,
    train_loader,
    val_loader,
    test_loader,
    task_index=0,
    out_path=None,
    bins=25,
    fig_scale=None,
    dpi=None,
    marker_size=28,
    marker_alpha=0.7,
    inverse_true=None,
    inverse_pred=None,
    kde_pad=0.05,
    palette=Palette(),
    rc=RcParamsConfig()):
    
    rc.apply()
    fig_cfg = FigureConfig(fig_scale=fig_scale)
    save_cfg = SaveConfig(
        out_path=out_path, dpi=dpi, suffix="_scatter"
        )
    x_tr, y_tr = pred_from_loader(
        model, train_loader, device,
        task_index, inverse_true, inverse_pred
        )
    x_va, y_va = pred_from_loader(
        model, val_loader, device,
        task_index, inverse_true, inverse_pred
        )
    x_te, y_te = pred_from_loader(
        model, test_loader, device,
        task_index, inverse_true, inverse_pred
        )
    x_all = (np.concatenate([x_tr, x_va, x_te])
             if x_tr.size + x_va.size + x_te.size
             else np.array([]))
    y_all = (np.concatenate([y_tr, y_va, y_te])
             if y_tr.size + y_va.size + y_te.size
             else np.array([]))
    if x_all.size == 0:
        raise ValueError("No valid pairs to plot.")
    lo = float(np.min([x_all.min(), y_all.min()]))
    hi = float(np.max([x_all.max(), y_all.max()]))
    if (not np.isfinite(lo) or not np.isfinite(hi)
        or hi == lo):
        lo, hi = 0.0, 1.0
    pad = kde_pad * (hi - lo if hi > lo else 1.0)
    fig = plt.figure(
        figsize=(fig_cfg.side(), fig_cfg.side())
        )
    gs = gridspec.GridSpec(
        nrows=2, ncols=2,
        width_ratios=[4, 1.1],
        height_ratios=[1.1, 4],
        wspace=0.02, hspace=0.02
        )
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    sets = [
        ("Train", x_tr, y_tr),
        ("Validation", x_va, y_va),
        ("Test", x_te, y_te),
        ]
    for (nm, xs, ys), c in zip(sets, palette.splits):
        ax.scatter(xs, ys, s=marker_size,
            alpha=marker_alpha, linewidths=0,
            label=nm, color=c
            )
    ax.plot([lo, hi], [lo, hi],
        linestyle="--", linewidth=1.0, color="black"
        )
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Experimental")
    ax.set_ylabel("Predicted")
    legend_frame(ax, fontsize=rc.legend_fontsize)
    style_spines(ax)
    ax.grid(False)
    ax.set_facecolor("white")
    for (nm, xs, _), c in zip(sets, palette.splits):
        ax_top.hist(xs, bins=bins, range=(
            lo - pad, hi + pad),
            color=c, alpha=0.6
            )
    ax_top.set_xlim(lo - pad, hi + pad)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_ylabel("Count")
    style_spines(ax_top)
    ax_top.grid(False)
    ax_top.set_facecolor("white")
    for (nm, _, ys), c in zip(sets, palette.splits):
        ax_right.hist(ys, bins=bins, range=(
            lo - pad, hi + pad),
            orientation="horizontal", 
            color=c, alpha=0.6
            )
    ax_right.set_ylim(lo - pad, hi + pad)
    ax_right.set_xlabel("Count")
    ax_right.tick_params(axis="y", labelleft=False)
    style_spines(ax_right)
    ax_right.grid(False)
    ax_right.set_facecolor("white")
    plt.tight_layout()
    save_cfg.maybe_save(fig)
    plt.show()


def visualize_logits(
    model,
    train_loader,
    val_loader,
    test_loader,
    task_index=0,
    bins=25,
    out_path=None,
    fig_scale=None,
    dpi=None,
    is_cls=None,
    inverse_true=None,
    inverse_pred=None,
    palette=Palette(),
    rc=RcParamsConfig()):

    _, labs = gather_labels(
        model, [train_loader, 
        val_loader, test_loader], device
        )
    kind = infer_task_kind(
        is_cls=is_cls, 
        task_index=task_index, y_true=labs
        )
    if kind == 'classification':
        histogram_plot(
            model,
            train_loader,
            val_loader,
            test_loader,
            task_index=task_index,
            out_path=out_path,
            bins=bins,
            fig_scale=fig_scale,
            dpi=dpi,
            palette=palette,
            rc=rc,
            )
    else:
        scatter_plot(
            model,
            train_loader,
            val_loader,
            test_loader,
            task_index=task_index,
            out_path=out_path,
            bins=bins,
            fig_scale=fig_scale,
            dpi=dpi,
            inverse_true=inverse_true,
            inverse_pred=inverse_pred,
            palette=palette,
            rc=rc,
            )