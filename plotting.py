import os
import matplotlib.pyplot as plt
from astropy.cosmology import FlatwCDM
from chainconsumer import ChainConsumer
import numpy as np
from scipy.interpolate import interp1d

plot_folder = "plots"
os.makedirs(plot_folder, exist_ok=True)


def plot(models, filename):
    c = ChainConsumer()
    for m in models:
        c.add_chain(m.samples, name=m.name, **m.kwargs)
    c.configure(bins=0.7)
    filename = os.path.join(plot_folder, filename)
    c.configure(legend_artists=True, flip=False)
    c.plotter.plot(filename=filename, truth=[0.3, -1, -19.36], figsize="COLUMN")
    c.plotter.plot(filename=filename.replace(".png", "_omw.png"), truth=[0.3, -1], parameters=[r"$\Omega_m$", "$w$"], figsize="COLUMN")
    # c.plotter.plot_walks(filename=filename.replace(".png", "_walks.png"), truth=[0.3, -1, -19.36])


def plot_hubble(models, filename):
    fig, axes = plt.subplots(figsize=(10, 8), nrows=2)

    zs = np.geomspace(0.01, 1.0, 500)
    mb_cosmo = FlatwCDM(70, 0.3, w0=-1).distmod(zs).value - 19.36
    interp = interp1d(zs, mb_cosmo)

    for m in models:
        ms = 3 if m.binned else 1
        axes[0].errorbar(m.zs, m.mbs, yerr=np.sqrt(np.diag(m.cov)), c=m.kwargs.get("color"), ms=ms, lw=0.5, fmt="o", label=m.name)
        axes[1].errorbar(m.zs, m.mbs - interp(m.zs), yerr=np.sqrt(np.diag(m.cov)), c=m.kwargs.get("color"), ms=ms, lw=0.5, fmt="o", label=m.name)

    axes[0].plot(zs, mb_cosmo, c="r", lw=1.0, label="Truth")
    axes[1].plot(zs, mb_cosmo - interp(zs), c="r", lw=1.0, label="Truth")

    axes[0].legend()
    fig.savefig(os.path.join(plot_folder, filename), bbox_inches="tight", dpi=200, transparent=True)
