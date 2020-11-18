import numpy as np
import pandas as pd
from astropy.cosmology import FlatwCDM
from scipy.stats import norm


def generate_default_data(n=1000, obs_err_scale=0.001, zmin=0.01, zmax=1.0, om=0.3, w=-1.0, seed=0, sigma_int=0.05, MB=-19.36, beta=3.1, H0=70):
    np.random.seed(seed + 1)
    zs = sorted(np.random.uniform(low=zmin, high=zmax, size=n))

    # stat_scale is how we can try and get contours directly on the truth value
    # So we can differentiate between stat fluct and systematics without
    # having to run 100s of simulations.
    MBs = norm(MB, obs_err_scale * sigma_int).rvs(n)
    cs = norm(0, 0.07).rvs(n)

    dist_mod = FlatwCDM(H0, om, w0=w).distmod(zs).value
    mb = dist_mod + MBs + beta * cs

    mb_err = 0.05 * np.ones(n)
    mb_obs = norm.rvs(mb, obs_err_scale * mb_err, n)

    c_err = 0.01 * np.ones(n)
    c_obs = norm.rvs(cs, obs_err_scale * c_err, n)

    return pd.DataFrame({"z": zs, "raw_mb_obs": mb_obs, "raw_mb_err": mb_err, "c_obs": c_obs, "c_err": c_err})


def calculate_corrected_mb_from_data(df, beta=3.1, sigma_int=0.05):
    # Emulating how BBC normally calculates a fiducial MU value
    df["mb"] = df["raw_mb_obs"] - beta * df["c_obs"]
    df["mb_err"] = np.sqrt(df["raw_mb_err"] ** 2 + sigma_int ** 2 + (beta * df["c_err"]) ** 2)
    return df


def weighted_avg_and_var(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return average, variance


def bin_data(df, zbins=20):
    zbins = np.linspace(df["z"].min(), df["z"].max() + 1e-6, zbins)
    indexes = np.digitize(df["z"], zbins)
    means, errs, zs = [], [], []

    # Subtract fiducial cosmo to linearise.
    # This is insensitive to cosmology, just 2d averaging only works on a line, no curves allowed
    cosmo = FlatwCDM(70, 0.3)
    fiducial = cosmo.distmod(df["z"]).value - 19.36

    for i in np.unique(indexes):
        mask = indexes == i
        weights = 1 / (df.loc[mask, "mb_err"] ** 2)
        mean, var = weighted_avg_and_var(df.loc[mask, "mb"] - fiducial[mask], weights)
        fiducial_mean, _ = weighted_avg_and_var(fiducial[mask], weights)
        mean_z, _ = weighted_avg_and_var(df.loc[mask, "z"], weights)
        zs.append(mean_z)
        means.append(mean + cosmo.distmod(mean_z).value - 19.36)
        errs.append(np.sqrt(1 / np.sum(weights)))

    return pd.DataFrame({"z": zs, "mb": means, "mb_err": errs})


def add_redshift_systematic(df):
    df = df.copy()

    def fn(zs):
        return FlatwCDM(70, 0.3, w0=-1.05).distmod(zs).value - FlatwCDM(70, 0.3, w0=-1.0).distmod(zs).value

    df["mb"] += fn(df["z"])
    return df


def add_beta_systematic(df, beta=3.4):
    return calculate_corrected_mb_from_data(df.copy(), beta=beta)


def compute_covariance_from_difference(df1, df2, scale=1.0):
    diff = scale * (df1["mb"] - df2["mb"])
    cov = diff[:, None] @ diff[None, :]
    return cov


if __name__ == "__main__":

    data_raw = generate_default_data()
    data_cor = calculate_corrected_mb_from_data(data_raw)
    data_bin = bin_data(data_cor)

    import matplotlib.pyplot as plt

    plt.errorbar(data_cor["zs"], data_cor["mb"], yerr=np.sqrt(data_cor["mb_err"]), c="#03befc", ms=1, lw=0.5, fmt="o")
    plt.errorbar(data_bin["zs"], data_bin["mb"], yerr=np.sqrt(data_bin["mb_err"]), c="k", fmt="o", ms=3)
    plt.show()
