import functools
from collections import namedtuple
from typing import Callable, Tuple

import jax
import jax.numpy as np

from rbig_jax.custom_types import InputData
from rbig_jax.information.entropy import histogram_entropy
from rbig_jax.information.rbig import TrainState, rbig_init
from rbig_jax.stopping import info_red_cond
from rbig_jax.transforms.block import (forward_gauss_block_transform,
                                       inverse_gauss_block_transform)
from rbig_jax.transforms.inversecdf import invgausscdf_forward_transform
from rbig_jax.transforms.linear import svd_transform


def get_tolerance_dimensions(n_samples: int) -> int:
    xxx = np.logspace(2, 8, 7)
    yyy = np.array([0.1571, 0.0468, 0.0145, 0.0046, 0.0014, 0.0001, 0.00001])
    tol_dimensions = np.interp(n_samples, xxx, yyy)
    return tol_dimensions


def information_reduction(
    X: np.ndarray, Y: np.ndarray, uni_entropy: Callable, tol_dims: int, p: float = 0.25,
) -> float:
    """calculates the information reduction between layers
    This function computes the multi-information (total correlation)
    reduction after a linear transformation.
    
    .. math::
        Y = XW \\
        II = I(X) - I(Y)
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Data before the transformation, where n_samples is the number
        of samples and n_features is the number of features.
    
    Y : np.ndarray, shape (n_samples, n_features)
        Data after the transformation, where n_samples is the number
        of samples and n_features is the number of features
        
    p : float, default=0.25
        Tolerance on the minimum multi-information difference
        
    Returns
    -------
    II : float
        The change in multi-information
        
    Information
    -----------
    Author: Valero Laparra
            Juan Emmanuel Johnson
    """
    # calculate the marginal entropy
    hx = jax.vmap(uni_entropy)(X.T)
    hy = jax.vmap(uni_entropy)(Y.T)

    # Information content
    delta_info = np.sum(hy) - np.sum(hx)
    tol_info = np.sqrt(np.sum((hy - hx) ** 2))

    # get tolerance
    n_dimensions = X.shape[1]

    # conditional
    cond = np.logical_or(
        tol_info < np.sqrt(n_dimensions * p * tol_dims ** 2), delta_info < 0
    )
    return np.array(np.where(cond, 0.0, delta_info))


def rbig_total_correlation(
    X_samples: InputData,
    marginal_uni: Callable,
    uni_entropy: Callable,
    n_iterations: int = 100,
    p: float = 0.25,
):

    # total correlation reduction
    tol_dims = get_tolerance_dimensions(X_samples.shape[0])

    total_corr_f = jax.partial(
        information_reduction, uni_entropy=uni_entropy, tol_dims=tol_dims, p=p,
    )

    marginal_uni_f_vectorized = jax.vmap(marginal_uni)
    # create function for scan

    def body(carry, inputs):

        # marginal gaussianization
        carry_trans = marginal_uni_f_vectorized(carry.T).T

        # inverse CDF transformation
        carry_trans = invgausscdf_forward_transform(carry_trans)

        # rotation
        carry_trans = svd_transform(carry_trans)

        # information reduction
        info_red = total_corr_f(carry, carry_trans)
        # print(type(info_red))

        return carry_trans, info_red

    return jax.lax.scan(f=body, init=X_samples, xs=None, length=n_iterations)
