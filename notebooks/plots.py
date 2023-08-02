import matplotlib.pyplot as plt
import numpy as np


def plot_zero_fraction(avg_counts, zero_fraction, **scatter_kwargs):
    # Compute the Poisson prediction
    Poiss_predicted_fraction = np.exp(-np.unique(avg_counts))

    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(avg_counts, zero_fraction, s=5, label="Data", **scatter_kwargs)
    ax.plot(
        np.unique(avg_counts),
        Poiss_predicted_fraction,
        c="red",
        label="Poisson Prediction",
    )

    ax.set_ylabel("Fraction of zero expression")
    ax.set_xlabel("Mean read counts")
    ax.set_title("Fraction of zeros for each gene vs Poisson prediction")

    ax.legend()
    ax.set_xscale("log")

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Exon Lengths")


def plot_depth_per_cell(depth, genes_detected):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Plot number of detected genes vs number of counts
    axs[0].scatter(depth, genes_detected, s=10)
    axs[0].set_xscale("log")
    axs[0].set_ylim(0, 20000)
    axs[0].set_ylabel("Number of detected Genes")
    axs[0].set_xlabel("Sum of counts")
    axs[0].set_title("")

    # Plot histogram of sequencing depths (1 pt)
    axs[1].hist(depth, bins=20)
    axs[1].set_xlabel("Sum of counts per cell")
    axs[1].set_ylabel("Number of Cells")
    axs[1].set_title("Sequencing depth per cell")


def plot_fano_factor(counts, ylim=None):
    mean_counts = counts.mean(axis=0)

    # Compute the variance of the expression counts of each gene
    var_counts = counts.var(axis=0)

    # Fano factor
    fano = var_counts / mean_counts

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].scatter(mean_counts, var_counts, s=10)
    axs[0].plot(
        np.arange(0, np.max(mean_counts), 1),
        np.arange(0, np.max(mean_counts), 1),
        color="r",
        linewidth=2,
        label="y = x",
    )

    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("Mean Expression")
    axs[0].set_ylabel("Expression Variance")
    axs[0].set_title("Gene variance vs mean expression")
    axs[0].tick_params(which="minor", length=1, color="k")
    axs[0].tick_params(which="major", length=5, color="k")
    axs[0].legend()

    # Fano Plot
    axs[1].scatter(mean_counts, fano, s=10)
    axs[1].plot(
        np.arange(0, np.max(mean_counts), 1),
        np.ones(np.arange(0, np.max(mean_counts), 1).size),
        color="r",
        linewidth=2,
        label="Model: F = 1",
    )
    axs[1].set_xlabel("Mean Expression")
    axs[1].set_ylabel("Fano Factor")
    axs[1].set_title("Fano Factor vs mean expression")
    if ylim is not None:
        axs[1].set_ylim(ylim[0], ylim[1])
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].tick_params(which="minor", length=2, color="k")
    axs[1].tick_params(which="major", length=5, color="k")

    axs[1].legend()


def plot_PCA_row(data_row, titles, cluster_colors):
    fig, axs = plt.subplots(1, len(data_row), figsize=(9, 3))

    for ax, data, title in zip(axs, data_row, titles):
        ax.scatter(data[:, 0], data[:, 1], s=3, c=cluster_colors, alpha=0.8)
        ax.set_title(title)

    fig.supxlabel("$1^{st}$ PC")
    fig.supylabel("$2^{nd}$ PC")


def plot_PCA_grid(pca_grid, selection_titles, norm_titles, cluster_colors):
    fig, axs = plt.subplots(
        len(pca_grid),
        len(pca_grid[0]),
        figsize=(3 * len(pca_grid[0]), 3 * len(pca_grid)),
    )

    for ax_row, data_row, selection_title in zip(axs, pca_grid, selection_titles):
        for ax, data, norm_title in zip(ax_row, data_row, norm_titles):
            ax.scatter(data[:, 0], data[:, 1], s=3, c=cluster_colors, alpha=0.8)
            ax.set_title(f"{norm_title} - {selection_title}")

    fig.supxlabel("$1^{st}$ PC")
    fig.supylabel("$2^{nd}$ PC")


def plot_TSNE(tsne_counts, cluster_colors, lims=None, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(tsne_counts[:, 0], tsne_counts[:, 1], s=20, c=cluster_colors, alpha=0.8)
    ax.set_title("TSNE")
    ax.set_axis_off()
    if lims is not None:
        ax.set_xlim(lims[0][0], lims[0][1])
        ax.set_ylim(lims[1][0], lims[1][1])
