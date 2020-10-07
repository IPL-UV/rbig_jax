from collections import namedtuple
from functools import partial
from rbig_jax.transforms.block import (
    forward_gauss_block_transform,
    inverse_gauss_block_transform,
)

from jax.scipy import stats
import tqdm
import jax
import jax.numpy as np

# from rbig_jax.transforms.gaussian import get_gauss_params
# from rbig_jax.transforms.linear import init_pca_params
from rbig_jax.transforms.histogram import get_hist_params
from rbig_jax.transforms.kde import get_kde_params
from rbig_jax.transforms.linear import compute_projection
from rbig_jax.transforms.marginal import (
    forward_gaussianization,
    forward_inversecdf,
    inverse_gaussianization,
)

RBIGParams = namedtuple(
    "RBIGParams", ["support_pdf", "empirical_pdf", "quantiles", "support", "rotation"]
)
# define gaussianization functions
forward_gauss = jax.jit(jax.vmap(forward_gaussianization))
inverse_gauss = jax.jit(jax.vmap(inverse_gaussianization))


def rbig_init(
    method: str = "histogram",
    support_ext: int = 10,
    precision: int = 50,
    alpha: float = 1e-5,
):
    """Initializes rbig function with fixed params
    
    Parameters
    ----------
    method : str
        the method used for marginal gaussianization
    support_extension : int
        the support extended for the domain when calculating
        the approximate pdf/cdfs for the marginal dists.
    precision: int
        the number of quantiles and approximate values
        
    Returns
    -------
    fit_func : Callable
        a callable function to fit the parameters of a new
        dataset
    
    Examples
    --------

    >>> fit_forward_func = rbig_init(
        method="histogram",
        support_extension=10,
        precision=1000,
        alpha=1e-5
        )
    >>> # Forwad function to fit
    >>> (
        Xtrans, ldX, 
        forward_func, inv_func
        ) = fit_forward_func(data)
    >>> # Forward function without fitting
    >>> Xtrans_, ldX_ = forward_transform(data)
    """
    if method == "histogram":
        forward_uniformization = jax.vmap(
            partial(
                get_hist_params,
                support_extension=support_ext,
                precision=precision,
                alpha=alpha,
            )
        )
    elif method == "kde":
        forward_uniformization = jax.vmap(
            partial(get_kde_params, support_extension=support_ext, precision=precision)
        )
    else:
        raise ValueError("Unrecognized method")

    # fit forward function
    def fit_forward(X):
        """"""
        # =========================
        # Marginal Uniformization
        # =========================
        X, log_det, uni_params = forward_uniformization(X.T)

        # transpose data
        X, log_det = X.T, log_det.T

        # =========================
        # Inverse CDF
        # =========================

        # clip boundaries
        X = np.clip(X, 1e-5, 1.0 - 1e-5)

        X = forward_inversecdf(X)

        # =========================
        # Log Determinant Jacobian
        # =========================
        log_det = log_det - jax.scipy.stats.norm.logpdf(X)

        # =========================
        # Rotation
        # =========================
        R = compute_projection(X)

        X = np.dot(X, R)

        params = RBIGParams(
            support_pdf=uni_params.support_pdf,
            empirical_pdf=uni_params.empirical_pdf,
            support=uni_params.support,
            quantiles=uni_params.quantiles,
            rotation=R,
        )

        return X, log_det, params

    return fit_forward


def forward_transform(params, X):

    # Marginal Gaussianization
    X, log_det = forward_gauss(X.T, params)
    X, log_det = X.T, log_det.T

    # Rotation
    X = np.dot(X, params.rotation)

    return X, log_det


def inverse_transform(params, X):

    # Rotation
    X = np.dot(X, params.rotation.T)

    # Marginal Gaussianization
    X = inverse_gauss(X.T, params)

    X = X.T
    return X

