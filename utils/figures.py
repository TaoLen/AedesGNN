import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from predictor import predict


class FigureConfig:
    def __init__(self, 
        fig_scale=1.2, 
        base_inch=3.5,
        base_mult=1.5):

        self.fig_scale = fig_scale
        self.base_inch = base_inch
        self.base_mult = base_mult

    def side(self):
        scale = self.fig_scale if self.fig_scale is not None else 1.0
        config = self.base_inch * scale * self.base_mult

        return config


class SaveConfig:
    def __init__(self, out_path=None, dpi=600, suffix=""):
        self.out_path = out_path
        self.dpi = dpi
        self.suffix = suffix

    def maybe_save(self, fig):
        if not self.out_path:
            return
        p = self.out_path
        has_ext = os.path.splitext(os.path.basename(p))[1] != ""
        if not has_ext:
            os.makedirs(p, exist_ok=True)
            fname = f"plot{self.suffix}.svg"
            full = os.path.join(p, fname)
        else:
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            full = f"{p}{self.suffix}.svg"
        fig.savefig(full, bbox_inches="tight", dpi=self.dpi)


class RcParamsConfig:
    def __init__(self,
        axes_linewidth=1.1,
        axes_labelsize=11,
        axes_titlesize=11,
        xtick_labelsize=10,
        ytick_labelsize=10,
        legend_fontsize=9,
        xtick_direction="out",
        ytick_direction="out",
        xtick_major_size=3.2,
        ytick_major_size=3.2,
        xtick_major_width=1.0,
        ytick_major_width=1.0):

        self.axes_linewidth = axes_linewidth
        self.axes_labelsize = axes_labelsize
        self.axes_titlesize = axes_titlesize
        self.xtick_labelsize = xtick_labelsize
        self.ytick_labelsize = ytick_labelsize
        self.legend_fontsize = legend_fontsize  
        self.xtick_direction = xtick_direction
        self.ytick_direction = ytick_direction
        self.xtick_major_size = xtick_major_size
        self.ytick_major_size = ytick_major_size
        self.xtick_major_width = xtick_major_width
        self.ytick_major_width = ytick_major_width

    def apply(self):
        plt.rcParams.update({
            "axes.linewidth": self.axes_linewidth,
            "axes.labelsize": self.axes_labelsize,
            "axes.titlesize": self.axes_titlesize,
            "xtick.labelsize": self.xtick_labelsize,
            "ytick.labelsize": self.ytick_labelsize,
            "legend.fontsize": self.legend_fontsize,   
            "xtick.direction": self.xtick_direction,
            "ytick.direction": self.ytick_direction,
            "xtick.major.size": self.xtick_major_size,
            "ytick.major.size": self.ytick_major_size,
            "xtick.major.width": self.xtick_major_width,
            "ytick.major.width": self.ytick_major_width
            })



class Palette:
    def __init__(
        self,
        inact="#756CF4",
        mid="#D9DCD9",
        act="#FC7777",
        splits=("#A1C3CE", "#335fe3", "#2AAD0F"),
        reg_cmap="bwr"):

        self.inact = inact
        self.mid = mid
        self.act = act
        self.splits = tuple(splits)
        self.reg_cmap = reg_cmap
        self.cls2 = {0: self.inact, 1: self.act}
        self.cls3 = {0: self.inact, 1: self.mid, 2: self.act}

    def color_for_binary(self, y):
        return self.cls2[int(bool(y))]

    def color_for_ternary(self, k):
        return self.cls3[int(k) % 3]


def to_2d(a):
    if torch.is_tensor(a):
        a = a.detach().cpu().numpy()
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[:, None]
    return a


def fix_orientation(y):
    cond = (y.ndim == 2 and y.shape[0] < y.shape[1]
            and y.shape[0] <= 16)
    return y.T if cond else y


def style_spines(ax):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.0)
        sp.set_color("black")


def legend_frame(ax_legend, fontsize=8):
    legend = ax_legend.legend(
        loc="upper left",
        frameon=True,
        fontsize=fontsize
        )
    fr = legend.get_frame()
    fr.set_facecolor("white")
    fr.set_edgecolor("black")
    fr.set_linewidth(1.0)

    return legend


def infer_task_kind(
    is_cls=None, 
    task_index=None, 
    y_true=None):

    if is_cls is not None and task_index is not None:
        try:
            return ('classification' if bool(
                is_cls[task_index])
                    else 'regression')
        except Exception:
            pass
    if y_true is not None and task_index is not None:
        col = y_true[:, task_index]
        col = col[np.isfinite(col)]
        if col.size > 0:
            uq = np.unique(col)
            if uq.size <= 10 and np.all(
                np.isin(uq, [0, 1])):
                return 'classification'
            
    return 'regression'


def gather_labels(model, loaders, device):
    y_pred_list, y_true_list = [], []
    for loader in loaders:
        if loader is None:
            continue
        y_pred, y_true, _ = predict(
            model, loader, device, 
            return_embeddings=False
            )
        y_pred_list.append(y_pred)
        y_true_list.append(y_true)
    y_pred_all = fix_orientation(
        to_2d(np.vstack(y_pred_list))
        )
    y_true_all = fix_orientation(
        to_2d(np.vstack(y_true_list))
        )
    
    return y_pred_all, y_true_all