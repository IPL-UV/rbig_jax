from collections import namedtuple
from functools import partial

import jax
import jax.numpy as np

from rbig_jax.transforms.histogram import histogram_transform
from rbig_jax.transforms.kde import kde_transform
from rbig_jax.transforms.linear import compute_projection
from rbig_jax.transforms.marginal import forward_inversecdf

TrainState = namedtuple(
    "TrainState",
    ["n_layers", "info_loss", "X"],  # number of layers  # information loss
)


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
                histogram_transform,
                support_extension=support_ext,
                precision=precision,
                alpha=alpha,
            )
        )
    elif method == "kde":
        forward_uniformization = jax.vmap(
            partial(kde_transform, support_extension=support_ext, precision=precision)
        )
    else:
        raise ValueError("Unrecognized method")

    # fit forward function
    def fit_forward(X):
        """"""
        # =========================
        # Marginal Uniformization
        # =========================
        X = forward_uniformization(X.T)

        # transpose data
        X = X.T

        # =========================
        # Inverse CDF
        # =========================

        # clip boundaries
        X = np.clip(X, 1e-5, 1.0 - 1e-5)

        X = forward_inversecdf(X)

        # =========================
        # Rotation
        # =========================
        R = compute_projection(X)

        X = np.dot(X, R)

        return X

    return fit_forward
