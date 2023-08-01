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


def plot_CTF(
    pca_CTF,
    pca_log_CTF,
    pca_sqrt_CTF,
    pca_norm_counts,
    pca_log2_norm_counts,
    pca_sqrt_norm_counts,
    cluster_colors,
):
    fig, axs = plt.subplots(2, 3, figsize=(9, 6))
    axs = axs.flatten()

    # Plot CTF

    axs[0].scatter(pca_CTF[:, 0], pca_CTF[:, 1], s=3, c=cluster_colors, alpha=0.8)
    axs[0].set_title("PCA")

    axs[1].scatter(
        pca_log_CTF[:, 0], pca_log_CTF[:, 1], s=3, c=cluster_colors, alpha=0.8
    )
    axs[1].set_title("PCA after $log_2(X+1)$")

    axs[2].scatter(
        pca_sqrt_CTF[:, 0], pca_sqrt_CTF[:, 1], s=3, c=cluster_colors, alpha=0.8
    )
    axs[2].set_title("PCA after sqrt")

    axs[3].scatter(
        pca_norm_counts[:, 0],
        pca_norm_counts[:, 1],
        s=3,
        c=cluster_colors,
        alpha=0.8,
    )
    axs[3].set_title("PCA")

    axs[4].scatter(
        pca_log2_norm_counts[:, 0],
        pca_log2_norm_counts[:, 1],
        s=3,
        c=cluster_colors,
        alpha=0.8,
    )
    axs[4].set_title("PCA after $log_2(X+1)$")
    axs[4].set_xlim(-50, 100)
    axs[4].set_ylim(-75, 75)

    axs[5].scatter(
        pca_sqrt_norm_counts[:, 0],
        pca_sqrt_norm_counts[:, 1],
        s=3,
        c=cluster_colors,
        alpha=0.8,
    )
    axs[5].set_title("PCA after sqrt")
    axs[5].set_xlim(-300, 300)
    axs[5].set_ylim(-200, 200)

    fig.supxlabel("$1^{st}$ PC")
    fig.supylabel("$2^{nd}$ PC")


def plot_TSNE(tsne_counts, cluster_colors):
    fig, axs = plt.subplots()
    axs.scatter(tsne_counts[:, 0], tsne_counts[:, 1], s=20, c=cluster_colors, alpha=0.8)
    axs.set_title("TSNE")
    axs.set_axis_off()
