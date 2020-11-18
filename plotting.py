import logging
import os
import matplotlib.pyplot as plt
from astropy.cosmology import FlatwCDM
from chainconsumer import ChainConsumer
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sb
from scipy.stats import binned_statistic

from config import cov_to_corr

plot_folder = "plots"
os.makedirs(plot_folder, exist_ok=True)


def plot(models, filename, **kwargs):
    c = ChainConsumer()
    for m in models:
        c.add_chain(m.samples, name=m.name, **m.kwargs)
    filename = os.path.join(plot_folder, filename)
    c.configure(bins=0.7, legend_artists=True, flip=False, **kwargs)
    c.configure_truth(color="k", ls=":", lw=0.5)
    c.plotter.plot(filename=filename, figsize="COLUMN", truth=[0.3, -1, -19.36])
    c.plotter.plot(filename=filename.replace(".png", "_omw.png"), parameters=[r"$\Omega_m$", "$w$"], figsize="COLUMN", truth=[0.3, -1, -19.36])


def plot_hubble(models, filename):
    fig, axes = plt.subplots(figsize=(10, 8), nrows=2)

    zs = np.geomspace(0.01, 1.0, 500)
    mb_cosmo = FlatwCDM(70, 0.3, w0=-1).distmod(zs).value - 19.36
    interp = interp1d(zs, mb_cosmo)

    for m in models:
        ms = 3 if m.binned else 1
        alpha = 1 if m.binned else 0.2
        axes[0].errorbar(m.zs, m.mbs, yerr=np.sqrt(np.diag(m.cov)), c=m.kwargs.get("color"), ms=ms, lw=0.5, fmt="o", label=m.name, alpha=alpha)
        axes[1].errorbar(m.zs, m.mbs - interp(m.zs), yerr=np.sqrt(np.diag(m.cov)), c=m.kwargs.get("color"), ms=ms, lw=0.5, fmt="o", label=m.name, alpha=alpha)

    axes[0].plot(zs, mb_cosmo, c="r", lw=1.0, label="Truth")
    axes[1].plot(zs, mb_cosmo - interp(zs), c="r", lw=1.0, label="Truth")

    axes[0].legend()
    fig.savefig(os.path.join(plot_folder, filename), bbox_inches="tight", dpi=200, transparent=True)


def plot_residuals(df, df_sys, filename):
    logging.info(f"Plotting resiuals to {filename}")
    fig, axes = plt.subplots(ncols=2, figsize=(10, 6))
    diff = df_sys["mb"] - df["mb"]

    for ax, p in zip(axes, ["c_obs", "z"]):
        ax.scatter(df_sys[p], diff, s=1, marker=".")
        mean, edges, _ = binned_statistic(df_sys[p], diff)
        edge_center = 0.5 * (edges[1:] + edges[:-1])
        ax.plot(edge_center, mean, c="k")

    fig.savefig(os.path.join(plot_folder, filename), bbox_inches="tight", dpi=200, transparent=True)
    logging.debug("Done plot")


def plot_cov(m):
    logging.debug(f"Plotting correlation for {m.name}")
    fig, ax = plt.subplots(figsize=(7, 5))
    cov = m.cov
    corr = cov_to_corr(cov)
    ax.imshow(corr, aspect="equal")
    ax.set_title(f"Correlation for {m.name}")
    fig.savefig(os.path.join(plot_folder, f"{m.name}_corr.png"), bbox_inches="tight", dpi=200, transparent=True)
