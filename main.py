from code.config import setup_logging
from code.fitter import SNIaModel, get_cosmo, SNIaKimModel
from code.generate_data import generate_default_data, standardise_data, add_cosmoredshift_systematic, add_betacolor_systematic
from code.plotting import plot, plot_hubble, plot_residuals, plot_cov
import logging


if __name__ == "__main__":
    setup_logging()

    # Generate raw data and the standardised data which has the beta*c correction applied
    data_raw = generate_default_data(n=2000)
    data_cor = standardise_data(data_raw)
    data_with_zsys = add_cosmoredshift_systematic(data_cor)
    data_with_betasys = add_betacolor_systematic(data_cor)

    models = [
        SNIaModel("Baseline, unbinned", data_cor, bin=False, color="#444444"),
        SNIaModel("Baseline, binned", data_cor, bin=True, color="#111111"),
        # Cov method
        SNIaModel("Redshift syst, unbinned", data_cor, data_with_zsys, bin=False, color="#444444"),  ##444444,#FB8C00
        SNIaModel("Redshift syst, binned", data_cor, data_with_zsys, bin=True, color="#111111"),
        SNIaModel("Color syst, unbinned", data_cor, data_with_betasys, bin=False, color="#444444"),  ##1976D2
        SNIaModel("Color syst, binned", data_cor, data_with_betasys, bin=True, color="#111111"),
        # Scale (Kim) method
        SNIaKimModel("Redshift syst, unbinned, scale", data_cor, data_with_zsys, bin=False, color="#444444"),
        SNIaKimModel("Redshift syst, binned, scale", data_cor, data_with_zsys, bin=True, color="#111111"),
        SNIaKimModel("Color syst, unbinned, scale", data_cor, data_with_betasys, bin=False, color="#444444"),
        SNIaKimModel("Color syst, binned, scale", data_cor, data_with_betasys, bin=True, color="#111111"),
    ]

    plot_hubble([m for m in models if "Baseline" in m.name], "hubble.png")
    plot_residuals(data_cor, data_with_betasys, "systematic_color.png")
    plot_residuals(data_cor, data_with_zsys, "systematic_redshift.png")

    logging.info("Starting model fits")
    for m in models:
        m.fit(steps=8000)

    scale = [m for m in models if "scale" in m.name]

    # Bulk plotting, here we come
    logging.info("Starting plotting")
    plot(models, "all_contours.png")
    plot([m for m in models if "Baseline" in m.name], "baseline_contours.png", shade=False, colors=["#87ceeb", "k"], flip=True)
    plot([m for m in models if "Color" in m.name and "scale" not in m.name], "color_contours.png", shade=False, colors=["k", "#999999"], flip=True)
    plot([m for m in models if "Redshift" in m.name and "scale" not in m.name], "redshift_contours.png", shade=False, colors=["k", "#999999"], flip=True)
    plot([m for m in models if "Color" in m.name and "scale" in m.name], "scale_color_contours.png", shade=False, colors=["k", "#999999"], flip=True)
    plot([m for m in models if "Redshift" in m.name and "scale" in m.name], "scale_redshift_contours.png", shade=False, colors=["k", "#999999"], flip=True)
    plot([m for m in models if "Color" in m.name], "all_color_contours.png", shade=False, colors=["b", "lb", "g", "lg"], flip=True)
    plot([m for m in models if "Redshift" in m.name], "all_redshift_contours.png", shade=False, colors=["b", "lb", "g", "lg"], flip=True)
    # plot_hubble([m for m in models if "Baseline" in m.name], "hubble.png")
    for m in models:
        plot_cov(m)
