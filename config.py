import logging
import numpy as np


def setup_logging():
    fmt = "[%(levelname)8s |%(filename)21s:%(lineno)3d]   %(message)s"
    # handler = logging.StreamHandler(sys.stdout)
    # logging.basicConfig(level=logging.DEBUG, format=fmt, handlers=[handler, logging.FileHandler("compute_h0.log")])
    logging.basicConfig(level=logging.DEBUG, format=fmt)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)


def cov_to_corr(cov):
    diag = np.sqrt(np.diag(cov))
    elem = diag[:, None] @ diag[None, :]
    return cov / elem
