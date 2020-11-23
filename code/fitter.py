from functools import lru_cache

import emcee
from scipy.interpolate import interp1d
from scipy.stats import norm, multivariate_normal as mv
from astropy.cosmology import FlatwCDM
import numpy as np
import logging
import pandas as pd
import os
from code.generate_data import bin_data, compute_covariance_from_difference


@lru_cache(maxsize=32768)
def get_cosmo(om, w):
    zs = np.geomspace(0.001, 1.2, 500)
    return interp1d(zs, FlatwCDM(70, om, w0=w).distmod(zs).value, kind="linear")


class SNIaModel:
    def __init__(self, name, *data, bin=False, add_cov=True, **kwargs):
        logging.info(f"Making class {name}")
        self.name = name
        self.binned = bin
        self.kwargs = {"linestyle": "--" if bin else "-"}
        self.kwargs.update(kwargs)
        self.chain_folder = "chains"
        self.chain_file = os.path.join(self.chain_folder, f"{name}.pkl")
        if bin:
            data = [bin_data(d) for d in data]
        self.datas = data

        # If we have multiple input data files, turn them into systematics
        self.data = data[0]
        self.zs = self.data["z"]
        self.mbs = self.data["mb"]
        self.cov = np.diag(self.data["mb_err"] ** 2)
        if len(data) > 1 and add_cov:
            for d in data[1:]:
                self.cov += compute_covariance_from_difference(self.data, d)
            self.norm = mv(self.mbs, cov=self.cov)
        else:
            self.norm = norm(self.mbs, np.sqrt(np.diag(self.cov)))
        self.samples = None

    def likelihood(self, om, w, MB, delta=None):
        # This is slow, calling quad over and over
        # cosmo = FlatwCDM(70, om, w0=w)
        # dist_mod = cosmo.distmod(self.zs).value

        # Using linear interp and lru_cache instead
        dist_mod = get_cosmo(np.round(om, 3), np.round(w, 3))(self.zs)
        pred = dist_mod + MB
        if delta is not None:
            pred -= delta
        return np.sum(self.norm.logpdf(pred))

    def posterior(self, x):
        om, w, MB = x
        if not 0.1 < om < 1.0 or not -2 < w < 0 or not -20 < MB < -18.5:
            return -np.inf
        like = self.likelihood(*x)
        if not np.isfinite(like):
            return -np.inf
        return like

    def fit(self, steps=500):
        os.makedirs(self.chain_folder, exist_ok=True)
        if os.path.exists(self.chain_file):
            logging.info(f"Reading existing chain file from {self.chain_file}")
            df = pd.read_pickle(self.chain_file)
            self.samples = df
            logging.info(f"Model {self.name} has {df.shape} samples")
            return df
        else:
            df = self.fit_model(steps=steps)
            logging.info(f"Saving chain file to {self.chain_file}")
            df.to_pickle(self.chain_file)
            self.samples = df
            return df

    def fit_model(self, steps=500, ndim=3, bounds=(0.1, -1.25, -19.6), cols=(r"$\Omega_m$", "$w$", "$M_B$")):
        nwalkers = 32
        p0 = 0.5 * np.random.uniform(size=(nwalkers, ndim)) + np.array(bounds)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.posterior)
        logging.debug(f"Running burn in for {self.name}")
        state = sampler.run_mcmc(p0, 100)
        sampler.reset()
        logging.debug(f"Sampling {self.name} for {steps} steps")
        sampler.run_mcmc(state, steps)
        samples = sampler.get_chain(flat=True)
        df = pd.DataFrame(samples, columns=cols)
        return df


class SNIaKimModel(SNIaModel):
    def __init__(self, name, *data, bin=False, scale_prior=1.0, **kwargs):
        super().__init__(name, *data, bin=bin, add_cov=False, **kwargs)
        assert len(data) == 2, "Only pass one syst for the toy model plz"
        self.delta = self.datas[1]["mb"] - self.datas[0]["mb"]
        self.scale_prior = scale_prior

    def posterior(self, x):
        om, w, MB, scale = x
        if not 0.1 < om < 1.0 or not -2 < w < 0 or not -20 < MB < -18.5:
            return -np.inf
        like = self.likelihood(om, w, MB, delta=scale * self.delta)
        if not np.isfinite(like):
            return -np.inf
        return like + norm.logpdf(scale, 0.0, self.scale_prior)

    def fit_model(self, steps=500, ndim=4, bounds=(0.1, -1.25, -19.6, -0.5), cols=(r"$\Omega_m$", "$w$", "$M_B$", "S")):
        return super().fit_model(steps=steps, ndim=ndim, bounds=bounds, cols=cols)
