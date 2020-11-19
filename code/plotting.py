import logging
import os
import matplotlib.pyplot as plt
from astropy.cosmology import FlatwCDM
from chainconsumer import ChainConsumer
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic
from code.config import cov_to_corr

plot_folder = "plots"
os.makedirs(plot_folder, exist_ok=True)


def plot(models, filename, **kwargs):
    kwargs["flip"] = kwargs.get("flip", False)
    c = ChainConsumer()
    for m in models:
        c.add_chain(m.samples, name=m.name, **m.kwargs)
    filename = os.path.join(plot_folder, filename)
    f2 = filename.replace(".png", "_omw.png")
    c.configure(bins=20, legend_artists=True, shade_gradient=0.0, **kwargs)
    c.plotter.plot(filename=[filename, filename.replace(".png", ".pdf")], figsize="COLUMN")
    c.plotter.plot(filename=[f2, f2.replace(".png", ".pdf")], parameters=[r"$\Omega_m$", "$w$"], figsize="COLUMN")


def plot_hubble(models, filename):
    plt.rcParams["text.usetex"] = True
    fig, axes = plt.subplots(figsize=(8, 4), nrows=2, gridspec_kw={"hspace": 0, "height_ratios": [2, 1]}, sharex=True)

    zs = np.geomspace(0.01, 1.0, 500)
    mb_cosmo = FlatwCDM(70, 0.3, w0=-1).distmod(zs).value - 19.36
    interp = interp1d(zs, mb_cosmo)

    for m in models:
        ms = 4 if m.binned else 1
        alpha = 1 if m.binned else 0.2
        ec = "#eb150e" if m.binned else "#cfc0a5"
        c = "#eb150e" if m.binned else "#e89c0e"
        axes[0].errorbar(m.zs, m.mbs, yerr=np.sqrt(np.diag(m.cov)), c=c, ms=ms, lw=0.5, ecolor=ec, fmt="o", label=m.name, alpha=alpha)
        axes[1].errorbar(m.zs, m.mbs - interp(m.zs), yerr=np.sqrt(np.diag(m.cov)), c=c, ms=ms, ecolor=ec, lw=0.5, fmt="o", label=m.name, alpha=alpha)

    axes[0].plot(zs, mb_cosmo, c="k", lw=1.2, label="Truth")
    axes[1].plot(zs, mb_cosmo - interp(zs), c="k", lw=1.0, label="Truth")
    axes[0].margins(x=0)
    axes[1].margins(x=0)
    axes[0].legend(frameon=False, loc=4)
    axes[1].set_xlabel("$z$")
    axes[0].set_ylabel("$m_B^*$")
    axes[1].set_ylabel("Residuals")

    path = os.path.join(plot_folder, filename)
    fig.savefig(path, bbox_inches="tight", dpi=200, transparent=True)
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", dpi=200, transparent=True)


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
