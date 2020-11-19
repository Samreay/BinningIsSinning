from config import setup_logging
from fitter import SNIaModel, get_cosmo
from generate_data import generate_default_data, add_redshift_systematic, add_color_systematic, calculate_corrected_mb_from_data
from plotting import plot, plot_hubble, plot_residuals, plot_cov
import logging


if __name__ == "__main__":
    setup_logging()

    # Generate raw data and the standardised data which has the beta*c correction applied
    data_raw = generate_default_data(n=2000)
    data_cor = calculate_corrected_mb_from_data(data_raw)
    data_with_zsys = add_redshift_systematic(data_cor)
    data_with_betasys = add_color_systematic(data_cor)

    # Shared options for plotting
    a = {"shade_gradient": 0.0}

    models = [
        SNIaModel("Baseline, unbinned", data_cor, bin=False, color="#444444", **a),
        SNIaModel("Baseline, binned", data_cor, bin=True, color="#111111", **a),
        SNIaModel("Redshift syst, unbinned", data_cor, data_with_zsys, bin=False, color="#FB8C00", **a),
        SNIaModel("Redshift syst, binned", data_cor, data_with_zsys, bin=True, color="#FB8C00", **a),
        SNIaModel("Color syst, unbinned", data_cor, data_with_betasys, bin=False, color="#1976D2", **a),
        SNIaModel("Color syst, binned", data_cor, data_with_betasys, bin=True, color="#1976D2", **a),
    ]

    logging.info("Starting model fits")
    for m in models:
        m.fit(steps=2000)
    print(get_cosmo.cache_info())

    # Bulk plotting, here we come
    logging.info("Starting plotting")
    plot(models, "all_contours.png")
    plot([m for m in models if "Color" in m.name], "color_contours.png", shade=False, colors=["p", "b"], flip=True)
    plot([m for m in models if "Redshift" in m.name], "redshift_contours.png", shade=False, colors=["a", "r"], flip=True)
    plot_hubble(models, "hubble.png")
    plot_residuals(data_cor, data_with_zsys, "systematic_redshift.png")
    plot_residuals(data_cor, data_with_betasys, "systematic_color.png")
    for m in models:
        plot_cov(m)
