from config import setup_logging
from fitter import SNIaModel, get_cosmo
from generate_data import generate_default_data, calculate_corrected_mb_from_data, add_redshift_systematic, add_beta_systematic
from plotting import plot, plot_hubble

if __name__ == "__main__":
    setup_logging()

    # Generate raw data and the standardised data which has the beta*c correction applied
    data_raw = generate_default_data(n=1000)
    data_cor = calculate_corrected_mb_from_data(data_raw)

    data_with_zsys = add_redshift_systematic(data_cor)
    data_with_betasys = add_beta_systematic(data_cor)

    a = {"shade_gradient": 0.0}

    models = [
        SNIaModel("Baseline, unbinned", data_cor, bin=False, color="#444444", **a),
        SNIaModel("Baseline, binned", data_cor, bin=True, color="#111111", **a),
        SNIaModel("Redshift syst, unbinned", data_cor, data_with_zsys, bin=False, color="#FB8C00", **a),
        SNIaModel("Redshift syst, binned", data_cor, data_with_zsys, bin=True, color="#FB8C00", **a),
        SNIaModel("Beta syst, unbinned", data_cor, data_with_betasys, bin=False, color="#1976D2", **a),
        SNIaModel("Beta syst, binned", data_cor, data_with_betasys, bin=True, color="#1976D2", **a),
    ]

    for m in models:
        m.fit()

    plot(models, "all_contours.png")
    plot_hubble(models, "hubble.png")
    print(get_cosmo.cache_info())
