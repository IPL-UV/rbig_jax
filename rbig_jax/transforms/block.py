from collections import namedtuple
from rbig_jax.transforms.histogram import hist_inverse_transform
from typing import Callable, Dict
import jax
import jax.numpy as np
from jax.util import partial
from rbig_jax.transforms.marginal import get_params_marginal, marginal_transform

from rbig_jax.transforms.gaussianize import (
    forward_gaussianize_transform,
    inverse_gaussianize_transform,
    get_gauss_params_hist,
    get_gauss_params_kde,
)
from rbig_jax.transforms.rotation import (
    get_pca_params,
    rot_forward_transform,
    rot_inverse_transform,
)

GaussParams = namedtuple(
    "GaussParams", ["rotation", "support", "quantiles", "support_pdf", "empirical_pdf"]
)


def init_gauss_hist_block_params(support_extension=10, precision=1_000, alpha=1e-5):

    # initialize hist function
    hist_init = partial(
        get_gauss_params_hist,
        support_extension=support_extension,
        precision=precision,
        alpha=alpha,
    )

    # initialize forward function
    init_func = partial(get_gauss_hist_block_params, hist_init=hist_init)
    return init_func


def init_gauss_kde_block_params(support_extension=10, precision=1_000):

    # initialize hist function
    kde_init = partial(
        get_gauss_params_kde, support_extension=support_extension, precision=precision,
    )

    # initialize forward function
    init_func = partial(get_gauss_hist_block_params, hist_init=kde_init)
    return init_func


def init_gauss_hist_block(support_extension=10, precision=1_000, alpha=1e-5):

    # initialize hist function
    hist_init = partial(
        get_gauss_params_hist,
        support_extension=support_extension,
        precision=precision,
        alpha=alpha,
    )

    # initialize forward function
    init_func = partial(get_gauss_hist_block, hist_init=hist_init)
    return init_func


def init_gauss_kde_block(support_extension=10, precision=1_000, alpha=1e-5):

    # initialize hist function
    kde_init = partial(
        get_gauss_params_kde,
        support_extension=support_extension,
        precision=precision,
        alpha=alpha,
    )

    # initialize forward function
    init_func = partial(get_gauss_hist_block, hist_init=kde_init)
    return init_func


def get_gauss_hist_block_params(X, hist_init: Callable):
    # Marginal Transformation
    X, mg_params = get_params_marginal(X, hist_init)

    # rotation
    X, rot_params = get_pca_params(X)

    # get params
    params = GaussParams(*rot_params, *mg_params)
    return X, params


def get_gauss_hist_block(X, hist_init: Callable):
    # Marginal Transformation
    X, _ = get_params_marginal(X, hist_init)

    # rotation
    X, _ = get_pca_params(X)
    return X


def forward_gauss_block_transform(X: np.ndarray, params: namedtuple):
    # Marginal Transformation
    X = marginal_transform(X, forward_gaussianize_transform, params)

    # rotation
    X = rot_forward_transform(X, params)

    return X


def inverse_gauss_block_transform(X: np.ndarray, params: namedtuple):

    # rotation
    X = rot_inverse_transform(X, params)

    # Marginal Transformation
    X = marginal_transform(X, inverse_gaussianize_transform, params)

    return X


def inverse_gauss_block_transform_constrained(X: np.ndarray, params: namedtuple):

    # rotation
    X = rot_inverse_transform(X, params)
    # forward transform
    X = init_gauss_hist_block()(X)
    # Marginal Transformation
    X = marginal_transform(X, inverse_gaussianize_transform, params)

    return X
