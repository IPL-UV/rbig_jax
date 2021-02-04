from typing import Callable, Union
from functools import partial

import jax
import jax.numpy as np

from rbig_jax.transforms.histogram import get_hist_params
from rbig_jax.transforms.inversecdf import (
    invgausscdf_inverse_transform,
    invgausscdf_forward_transform,
)

from rbig_jax.transforms.kde import get_kde_params
from rbig_jax.transforms.uniformize import (
    uniformize_transform,
    uniformize_inverse,
    uniformize_gradient,
)
from rbig_jax.transforms.marginal import marginal_transform, marginal_transform_gradient


def gaussianize_forward(
    X: np.ndarray, uni_transform_f: Callable, return_params: bool = True
):
    """Gaussianization Transformation w. Params"""
    # forward uniformization function
    X, params = uni_transform_f(X)
    # clip boundaries
    X = np.clip(X, 1e-5, 1.0 - 1e-5)

    # inverse cdf
    X = invgausscdf_forward_transform(X)

    return X, params


def gaussianize_transform(X: np.ndarray, params, return_jacobian=True):

    # forward uniformization function
    X = uniformize_transform(X, params)

    # clip boundaries
    X = np.clip(X, 1e-5, 1.0 - 1e-5)

    # inverse cdf
    X = invgausscdf_forward_transform(X)

    return X


def gaussianize_marginal_transform(X: np.ndarray, params):

    # forward uniformization function
    X = marginal_transform(X, uniformize_transform, params)

    # clip boundaries
    X = np.clip(X, 1e-5, 1.0 - 1e-5)

    # inverse cdf
    X = invgausscdf_forward_transform(X)

    return X


def gaussianize_marginal_gradient(X: np.ndarray, params):

    # Log PDF of uniformized data
    Xu_dj = marginal_transform(X, uniformize_gradient, params)

    Xu_ldj = np.log(Xu_dj)

    # forward uniformization function
    X = marginal_transform(X, uniformize_transform, params)

    # clip boundaries
    X = np.clip(X, 1e-5, 1.0 - 1e-5)

    # inverse cdf
    X = invgausscdf_forward_transform(X)

    # Log PDF for Gaussianized data
    Xg_ldj = jax.scipy.stats.norm.ppf(X)

    # Full log transformation
    X_ldj = Xu_ldj - Xg_ldj

    return X, X_ldj


def gaussianize_inverse(X: np.ndarray, params):

    # inverse cdf
    X = invgausscdf_inverse_transform(X)

    # inverse  uniformization function
    X = marginal_transform(X, uniformize_inverse, params)

    return X


def gaussianize_marginal_inverse(X: np.ndarray, params):

    # inverse cdf
    X = invgausscdf_inverse_transform(X)

    # inverse  uniformization function
    X = marginal_transform(X, uniformize_inverse, params)

    return X


# TODO: Implement better clipping scheme for transformations

# def init_params_hist_1d(support_extension=10, precision=100, alpha=1e-5):

#     param_getter = jax.jit(
#         partial(
#             get_params,
#             support_extension=support_extension,
#             precision=precision,
#             alpha=alpha,
#         )
#     )

#     return param_getter


# def init_params(support_extension=10, precision=100, alpha=1e-5, method="histogram"):
#     if method == "histogram":
#         param_getter = jax.jit(
#             jax.vmap(
#                 partial(
#                     get_hist_params,
#                     support_extension=support_extension,
#                     precision=precision,
#                     alpha=alpha,
#                 )
#             )
#         )
#     elif method == "kde":
#         param_getter = jax.jit(
#             jax.vmap(
#                 partial(
#                     get_kde_params,
#                     support_extension=support_extension,
#                     precision=precision,
#                 )
#             )
#         )
#     else:
#         raise ValueError(f"Unrecognized method...")
#     return param_getter


def get_gauss_params_hist(X, support_extension=10, precision=1000, alpha=1e-5):

    X, params = get_hist_params(
        X, support_extension=support_extension, precision=precision, alpha=alpha
    )
    # clip boundaries
    X = np.clip(X, 1e-5, 1.0 - 1e-5)

    # inverse cdf
    X = invgausscdf_forward_transform(X)

    return X, params


def get_gauss_params_kde(X, support_extension=10, precision=1000, alpha=1e-5):

    X, params = get_kde_params(
        X, support_extension=support_extension, precision=precision
    )
    # clip boundaries
    X = np.clip(X, 1e-5, 1.0 - 1e-5)

    # inverse cdf
    X = invgausscdf_forward_transform(X)

    return X, params


def forward_gaussianize_transform(X, params):

    # Unformization transformation
    X = uniformize_transform(X, params)

    # clip boundaries
    X = np.clip(X, 1e-5, 1.0 - 1e-5)

    # inverse cdf transformation
    X = invgausscdf_forward_transform(X)
    return X


def inverse_gaussianize_transform(X, params):

    X = invgausscdf_inverse_transform(X)

    # X = np.clip(X, 1e-5, 1.0 - 1e-5)
    X = uniformize_inverse(X, params)

    return X


def inverse_gaussianize_transform_constrained(X, params, func: Callable):

    X, _ = func(X)

    X = invgausscdf_inverse_transform(X)

    X = uniformize_inverse(X, params)

    return X


# def get_gauss_params(X, apply_func):
#     X, ldX, params = apply_func(X)

#     # clip boundaries
#     X = np.clip(X, 1e-5, 1.0 - 1e-5)

#     X = forward_inversecdf(X)

#     log_prob = ldX - jax.scipy.stats.norm.logpdf(X)

#     return None


# def mg_forward_transform(X, params):

#     X = hist_forward_transform(X, params)
#     X = invgauss_forward_transform(X)
#     return X


# def mg_inverse_transform(X, params):

#     X = invgauss_inverse_transform(X)

#     X = hist_inverse_transform(X, params)

#     return X


# def mg_gradient_transform():
#     return None


# def get_gauss_params_1d(X, apply_func):
#     X, ldX, params = apply_func(X)

#     # clip boundaries
#     X = np.clip(X, 1e-5, 1.0 - 1e-5)

#     X = forward_inversecdf_1d(X)

#     log_prob = ldX - jax.scipy.stats.norm.logpdf(X)

#     return (
#         X,
#         log_prob,
#         params,
#         forward_gaussianization,
#         inverse_gaussianization,
#     )


# @jax.jit
# def forward_gaussianization(X, params):

#     # transform to uniform domain
#     X, Xdj = forward_uniformization(X, params)

#     # clip boundaries
#     X = np.clip(X, 1e-5, 1.0 - 1e-5)

#     # transform to the gaussian domain
#     X = forward_inversecdf(X)

#     log_prob = Xdj - jax.scipy.stats.norm.logpdf(X)

#     return X, log_prob
