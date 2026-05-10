import numpy as np
from typing import Optional, Any
from matplotlib import pyplot as plt
from numpy import ndarray, dtype


def plt_save_np_hist(hist, bins, title, xlable, ylable, name):
    centers = (bins[:-1] + bins[1:]) / 2
    plt.figure(figsize=(6, 4))
    plt.bar(centers, hist, width=bins[1] - bins[0], edgecolor='black', alpha=0.7)
    plt.yscale('log')
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.title(title)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(name, dpi=150, bbox_inches='tight')
    plt.close()
    return


def mask_error_histogram(pred_awt: np.ndarray,
                         gt_awt: np.ndarray,
                         valid: Optional[np.ndarray, slice] = None,
                         bins: np.ndarray = None) -> dict[str, Any]:
    """
    The one returns hist of [pred - gt]
    """
    if bins is None:
        bins = np.linspace(-10, 10, 201)

    if valid is None:
        valid = (pred_awt > 0) & (gt_awt > 0)

    abs_err = (pred_awt[valid] - gt_awt[valid])
    hist, edges = np.histogram(abs_err, bins=bins, density=False)
    return dict(
        hist=hist,
        edges=edges,
        av_err=abs_err.mean(),
        std=abs_err.std(),
        mae=np.abs(abs_err).mean()
    )
