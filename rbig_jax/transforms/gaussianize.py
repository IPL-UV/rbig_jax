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
    forward_uniformization,
    inverse_uniformization,
)

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
    X = forward_uniformization(X, params)

    # clip boundaries
    X = np.clip(X, 1e-5, 1.0 - 1e-5)

    # inverse cdf transformation
    X = invgausscdf_forward_transform(X)
    return X


def inverse_gaussianize_transform(X, params):

    X = invgausscdf_inverse_transform(X)

    # X = np.clip(X, 1e-5, 1.0 - 1e-5)
    X = inverse_uniformization(X, params)

    return X


def inverse_gaussianize_transform_constrained(X, params, func: Callable):

    X, _ = func(X)

    X = invgausscdf_inverse_transform(X)

    X = inverse_uniformization(X, params)

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
