"""True-vs-predicted scatter plotting (lifted unchanged from true_pred_plots.py)."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_scatter(true_vals, pred_vals, outpath, title, xlim, ylim, xlabel, ylabel,
                 xscale='linear', yscale='linear', eps=1e-8):
    """Create scatter plot comparing true vs predicted values.

    Args:
        true_vals: Array of true values
        pred_vals: Array of predicted values
        outpath: Output file path
        title: Plot title
        xlim: X-axis limits
        ylim: Y-axis limits
        xlabel: X-axis label
        ylabel: Y-axis label
        xscale: 'linear' or 'log'
        yscale: 'linear' or 'log'
        eps: Clipping value for log scales
    """
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")

    # Clip to positive values for log scales
    true_vals_plot = np.clip(true_vals, eps, None) if xscale == 'log' else true_vals
    pred_vals_plot = np.clip(pred_vals, eps, None) if yscale == 'log' else pred_vals

    plt.figure(figsize=(6, 6), dpi=300)
    plt.scatter(true_vals_plot, pred_vals_plot, alpha=0.4, edgecolor='k', s=7)

    # Plot 1:1 reference line
    if xscale == 'log' and yscale == 'log':
        plt.plot(xlim, ylim, linestyle="--", color="red", linewidth=1,
                label="Perfect Prediction (1:1)")
    else:
        plt.plot([0, max(xlim)], [0, max(ylim)], linestyle="--", color="red",
                linewidth=1, label="Perfect Prediction (1:1)")

    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=16)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc="upper left")
    plt.grid(True, linewidth=1, alpha=1, which='minor')
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
