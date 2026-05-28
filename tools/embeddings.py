import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap.umap_ import UMAP

from params import load_embeddings
from figures import (
    FigureConfig,
    SaveConfig,
    RcParamsConfig,
    Palette,
    style_spines,
    legend_frame
    )


def normalize_embeddings(embeddings, jitter_std=0.0, seed=None):
    X = np.asarray(embeddings, dtype=float)
    if jitter_std > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(
            loc=0.0, 
            scale=jitter_std, 
            size=X.shape
        )
        X = X + noise
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n



def embeddings2tSNE(
    embeddings, 
    perplexity,
    learning_rate, 
    seed):

    tsne = TSNE(n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=1000,
        random_state=seed,
        init='pca'
        )
    dimension = tsne.fit_transform(embeddings)

    return dimension


def embeddings2uMAP(
    embeddings,
    n_neighbors, 
    min_dist, 
    seed):

    umap = UMAP(n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed
        )
    dimension = umap.fit_transform(embeddings)

    return dimension


def plot_embeddings(
    reduced,
    labels,
    file_path,
    title,
    x_label,
    y_label,
    cls_type=None,
    palette=Palette(),
    rc=RcParamsConfig(),
    save_cfg=None,
    bins=30,
    fig_scale=None,
    dpi=600,
    height_scale=0.66):

    rc.apply()
    plt.rcParams.update({
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": max(rc.legend_fontsize, 10),
    })
    if save_cfg is None:
        save_cfg = SaveConfig(
            out_path=file_path,
            dpi=dpi, suffix="_embed")

    fig_cfg = FigureConfig(fig_scale=fig_scale)
    side = fig_cfg.side()
    fig_width = side * 0.765
    fig_height = side * height_scale
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)
    lab = (labels.numpy() if torch.is_tensor(labels)
           else labels)
    lab = np.asarray(lab).squeeze()
    if lab.ndim > 1:
        lab = lab.reshape(-1)
    lab_float = lab.astype(float, copy=False)
    valid = ~np.isnan(lab_float)
    x = reduced[:, 0]
    y = reduced[:, 1]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if all(np.isfinite(v) for v in (xmin, xmax, ymin, ymax)):
        x_span = max(xmax - xmin, 1e-9)
        y_span = max(ymax - ymin, 1e-9)
        x_pad = 0.08 * x_span
        y_pad = 0.08 * y_span
        xlo = xmin - x_pad
        xhi = xmax + x_pad
        ylo = ymin - y_pad
        yhi = ymax + y_pad
    else:
        xlo, xhi = 0.0, 1.0
        ylo, yhi = 0.0, 1.0
    if cls_type is True:
        if lab.dtype.kind in ("f", "c"):
            bin_lab = (lab_float >= 0.5).astype(int)
        else:
            bin_lab = lab.astype(int, copy=False)

        ax.scatter(x[~valid], y[~valid],
            c="lightgray", s=6, alpha=0.4
            )
        bin_valid = bin_lab[valid]
        colors = np.where(
            bin_valid == 0,
            palette.inact,
            palette.act
            )
        ax.scatter(x[valid], y[valid], c=colors,
            s=9, alpha=0.8
            )
        hdl = [plt.Line2D(
                [0], [0], marker="o",
                linestyle="", label="Inactive",
                markerfacecolor=palette.inact,
                markeredgecolor="none",
                markersize=6, alpha=0.8),
            plt.Line2D([0], [0], marker="o",
                linestyle="", label="Active",
                markerfacecolor=palette.act,
                markeredgecolor="none",
                markersize=6, alpha=0.8),
                ]
        leg = ax.legend(handles=hdl,
            loc="upper left", frameon=True,
            fontsize=max(rc.legend_fontsize, 10)
            )
        fr = leg.get_frame()
        fr.set_facecolor("white")
        fr.set_edgecolor("black")
        fr.set_linewidth(1.0)

    elif cls_type is False:
        ax.scatter(x[~valid], y[~valid],
            c="lightgray", s=6, alpha=0.4
            )
        sc = ax.scatter(x[valid], y[valid],
            c=lab_float[valid], s=9, alpha=0.8,
            cmap=palette.reg_cmap
            )
        cb = fig.colorbar(sc, ax=ax,
            location="right", fraction=0.05,
            pad=0.03, use_gridspec=True
            )
        cb.ax.yaxis.set_label_position("right")
        cb.ax.yaxis.set_ticks_position("right")
        cb.ax.tick_params(labelsize=max(rc.legend_fontsize, 12))
        cb.set_label("Label")
    else:
        ax.scatter(x, y, c="gray", s=9, alpha=0.8)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect("auto")
    ax.set_xlabel("")
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_xlabel(x_label, fontsize=10)
    style_spines(ax)
    ax.grid(False)
    fig.set_size_inches(fig_width, fig_height, forward=True)
    plt.tight_layout()
    if save_cfg is not None:
        old_out_path = save_cfg.out_path
        old_dpi = save_cfg.dpi
        old_suffix = save_cfg.suffix
        if old_out_path:
            import os
            p = old_out_path
            has_ext = os.path.splitext(os.path.basename(p))[1] != ""
            if not has_ext:
                os.makedirs(p, exist_ok=True)
                full = os.path.join(p, f"plot{old_suffix}.svg")
            else:
                os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
                full = f"{p}{old_suffix}.svg"
            fig.savefig(full, dpi=old_dpi)
    plt.show()


def visualize_embeddings(
    in_path,
    epoch,
    method,
    task_index=None,
    out_path=None,
    perplexity=30,
    learning_rate=200,
    n_neighbors=15,
    min_dist=0.1,
    seed=42,
    fig_scale=None,
    dpi=600,
    palette=Palette(),
    rc=RcParamsConfig(),
    jitter_std=0.0):

    loaded = load_embeddings(in_path, epoch)
    if len(loaded) == 3:
        embeddings, labels, is_cls = loaded
    else:
        embeddings, labels = loaded
        is_cls = None
    if isinstance(embeddings, list):
        embeddings = torch.cat(embeddings, dim=0)
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)
    if torch.is_tensor(labels) and labels.dim() > 1:
        if task_index is None:
            raise ValueError("Please specify task_index")
        labels = labels[:, task_index]
    labels_np = (labels.tolist()
                 if torch.is_tensor(labels)
                 else labels)
    emb = embeddings.detach().cpu().numpy()
    emb = normalize_embeddings(
        emb,
        jitter_std=jitter_std,
        seed=seed,
        )

    if method == "tSNE":
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=1000,
            random_state=seed,
            init="pca")
        reduced = reducer.fit_transform(emb)
    elif method == "uMAP":
        reducer = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=seed)
        reduced = reducer.fit_transform(emb)
    else:
        raise ValueError(
            "Method must be 'tSNE' or 'uMAP'")
    cls_type = None
    if is_cls is not None and task_index is not None:
        cls_type = (
            bool(is_cls[task_index].item())
            if torch.is_tensor(is_cls)
            else bool(is_cls[task_index]))
    plot_embeddings(
        reduced=reduced,
        labels=labels_np,
        file_path=out_path,
        title=(f"{method} Embedding "
               f"(Epoch {epoch+1})"),
        x_label=(f"{method} Dim 1"),
        y_label=(f"{method} Dim 2"),
        cls_type=cls_type, palette=palette, rc=rc,
        save_cfg=SaveConfig(
            out_path=out_path,
            dpi=dpi,
            suffix="_embed",),
        fig_scale=fig_scale,
        dpi=dpi)
