from collections import namedtuple
from functools import partial
from typing import Callable

import jax
import jax.numpy as np
from jax.scipy import stats

from rbig_jax.transforms.gaussianize import (gaussianize_marginal_gradient,
                                             gaussianize_marginal_inverse,
                                             gaussianize_marginal_transform)
from rbig_jax.transforms.linear import compute_projection_v1

RBIGParams = namedtuple(
    "RBIGParams", ["support_pdf", "empirical_pdf", "quantiles", "support", "rotation"]
)
# define gaussianization functions
# forward_gauss = jax.jit(jax.vmap(forward_gaussianization))
# inverse_gauss = jax.jit(jax.vmap(inverse_gaussianization))


# def rbig_init(
#     method: str = "histogram",
#     support_ext: int = 10,
#     precision: int = 50,
#     alpha: float = 1e-5,
# ):
#     """Initializes rbig function with fixed params

#     Parameters
#     ----------
#     method : str
#         the method used for marginal gaussianization
#     support_extension : int
#         the support extended for the domain when calculating
#         the approximate pdf/cdfs for the marginal dists.
#     precision: int
#         the number of quantiles and approximate values

#     Returns
#     -------
#     fit_func : Callable
#         a callable function to fit the parameters of a new
#         dataset

#     Examples
#     --------

#     >>> fit_forward_func = rbig_init(
#         method="histogram",
#         support_extension=10,
#         precision=1000,
#         alpha=1e-5
#         )
#     >>> # Forwad function to fit
#     >>> (
#         Xtrans, ldX,
#         forward_func, inv_func
#         ) = fit_forward_func(data)
#     >>> # Forward function without fitting
#     >>> Xtrans_, ldX_ = forward_transform(data)
#     """
#     if method == "histogram":
#         forward_uniformization = jax.vmap(
#             partial(
#                 get_hist_params,
#                 support_extension=support_ext,
#                 precision=precision,
#                 alpha=alpha,
#             )
#         )
#     elif method == "kde":
#         forward_uniformization = jax.vmap(
#             partial(get_kde_params, support_extension=support_ext, precision=precision)
#         )
#     else:
#         raise ValueError("Unrecognized method")

#     # fit forward function
#     def fit_forward(X):
#         """"""
#         # =========================
#         # Marginal Uniformization
#         # =========================
#         X, log_det, uni_params = forward_uniformization(X.T)

#         # transpose data
#         X, log_det = X.T, log_det.T

#         # =========================
#         # Inverse CDF
#         # =========================

#         # clip boundaries
#         X = np.clip(X, 1e-5, 1.0 - 1e-5)

#         X = forward_inversecdf(X)

#         # =========================
#         # Log Determinant Jacobian
#         # =========================
#         log_det = log_det - jax.scipy.stats.norm.logpdf(X)

#         # =========================
#         # Rotation
#         # =========================
#         R = compute_projection(X)

#         X = np.dot(X, R)

#         params = RBIGParams(
#             support_pdf=uni_params.support_pdf,
#             empirical_pdf=uni_params.empirical_pdf,
#             support=uni_params.support,
#             quantiles=uni_params.quantiles,
#             rotation=R,
#         )

#         return X, log_det, params

#     return fit_forward


def rbig_block_forward(X, marginal_gauss_f: Callable):

    # ===================================
    # MARGINAL GAUSSIANIZATION
    # ===================================
    X, uni_params = marginal_gauss_f(X)

    # =========================
    # Rotation
    # =========================
    R = compute_projection_v1(X)

    X = np.dot(X, R)

    # save parameters
    params = RBIGParams(
        support_pdf=uni_params.support_pdf,
        empirical_pdf=uni_params.empirical_pdf,
        support=uni_params.support,
        quantiles=uni_params.quantiles,
        rotation=R,
    )
    return X, params


def rbig_block_transform(X, params):

    # ===================================
    # MARGINAL GAUSSIANIZATION
    # ===================================
    X = gaussianize_marginal_transform(X, params)

    # =========================
    # Rotation
    # =========================
    X = np.dot(X, params.rotation)

    return X


def rbig_block_transform_gradient(X, params):

    # ===================================
    # MARGINAL GAUSSIANIZATION
    # ===================================
    X, ldj = gaussianize_marginal_gradient(X, params)

    # =========================
    # Rotation
    # =========================
    X = np.dot(X, params.rotation)

    return X, ldj


def rbig_block_inverse(X, params):

    # =========================
    # Rotation
    # =========================
    X = np.dot(X, params.rotation.T)

    # ===================================
    # MARGINAL GAUSSIANIZATION
    # ===================================
    X = gaussianize_marginal_inverse(X, params)

    return X


# def forward_transform(params, X):

#     # Marginal Gaussianization
#     X, log_det = forward_gauss(X.T, params)
#     X, log_det = X.T, log_det.T

#     # Rotation
#     X = np.dot(X, params.rotation)

#     return X, log_det


# def inverse_transform(params, X):

#     # Rotation
#     X = np.dot(X, params.rotation.T)

#     # Marginal Gaussianization
#     X = inverse_gauss(X.T, params)

#     X = X.T
#     return X
