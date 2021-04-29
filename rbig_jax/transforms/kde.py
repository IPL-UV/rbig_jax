from functools import partial
from typing import Callable, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp
from chex import Array, dataclass

from rbig_jax.transforms.base import InitFunctions, InitLayersFunctions
from rbig_jax.transforms.marginal import MarginalUniformizeTransform
from rbig_jax.utils import get_domain_extension


class UniKDEParams(NamedTuple):
    support: Array
    quantiles: Array
    support_pdf: Array
    empirical_pdf: Array


def InitUniKDETransform(
    shape: Tuple,
    support_extension: Union[int, float] = 10,
    precision: int = 1_000,
    bw: float = 0.1,
    jitted: bool = False,
):

    n_samples, n_features = shape

    bw = estimate_bw(n_samples, n_features, method=bw)

    f = jax.partial(
        init_kde_params,
        support_extension=support_extension,
        precision=precision,
        bw=bw,
        return_params=True,
    )

    f_slim = jax.partial(
        init_kde_params,
        support_extension=support_extension,
        precision=precision,
        bw=bw,
        return_params=False,
    )
    if jitted:
        f = jax.jit(f)
        f_slim = jax.jit(f_slim)

    def params_and_transform(inputs, **kwargs):

        outputs, params = jax.vmap(f, out_axes=(1, 0), in_axes=(1,))(inputs)
        return outputs, params

    def init_params(inputs, **kwargs):

        _, params = jax.vmap(f, out_axes=(1, 0), in_axes=(1,))(inputs)
        return params

    def transform(inputs, **kwargs):

        outputs = jax.vmap(f_slim, out_axes=1, in_axes=(1,))(inputs)
        return outputs

    def bijector_and_transform(inputs, **kwargs):
        outputs, params = jax.vmap(f, out_axes=(1, 0), in_axes=(1,))(inputs)

        bijector = MarginalUniformizeTransform(
            support=params.support,
            quantiles=params.quantiles,
            support_pdf=params.support_pdf,
            empirical_pdf=params.empirical_pdf,
        )
        return outputs, bijector

    def bijector(X, **kwargs):
        _, params = jax.vmap(f, out_axes=(1, 0), in_axes=(1,))(X)

        bijector = MarginalUniformizeTransform(
            support=params.support,
            quantiles=params.quantiles,
            support_pdf=params.support_pdf,
            empirical_pdf=params.empirical_pdf,
        )
        return bijector

    return InitLayersFunctions(
        bijector=bijector,
        bijector_and_transform=bijector_and_transform,
        transform=transform,
        params=init_params,
        params_and_transform=params_and_transform,
    )


def init_kde_params(
    X: jnp.ndarray,
    bw: float = 0.1,
    support_extension: Union[int, float] = 10,
    precision: int = 1_000,
    return_params: bool = True,
):
    # generate support points
    lb, ub = get_domain_extension(X, support_extension)
    support = jnp.linspace(lb, ub, precision)

    # calculate the pdf for gaussian pdf
    pdf_support = broadcast_kde_pdf(support, X, bw)

    # calculate the cdf for support points
    factor = normalization_factor(X, bw)

    quantiles = broadcast_kde_cdf(support, X, factor)

    # forward transformation
    outputs = jnp.interp(X, support, quantiles)

    if return_params is True:

        # initialize parameters
        params = UniKDEParams(
            support=support,
            quantiles=quantiles,
            support_pdf=support,
            empirical_pdf=pdf_support,
        )

        return outputs, params
    else:
        return outputs


def kde_transform(
    X: jnp.ndarray, support_extension: Union[int, float] = 10, precision: int = 1_000,
):
    # generate support points
    lb, ub = get_domain_extension(X, support_extension)
    grid = jnp.linspace(lb, ub, precision)

    bw = scotts_method(X.shape[0], 1) * 0.5

    # calculate the cdf for grid points
    factor = normalization_factor(X, bw)

    x_cdf = broadcast_kde_cdf(grid, X, factor)

    X_transform = jnp.interp(X, grid, x_cdf)

    return X_transform


def broadcast_kde_pdf(eval_points, samples, bandwidth):

    n_samples = samples.shape[0]
    # print(n_samples, bandwidth)

    # distances (use broadcasting)
    rescaled_x = (
        eval_points[:, jnp.newaxis] - samples[jnp.newaxis, :]
    ) / bandwidth  # (2 * bandwidth ** 2)
    # print(rescaled_x.shape)
    # compute the gaussian kernel
    gaussian_kernel = 1 / jnp.sqrt(2 * jnp.pi) * jnp.exp(-0.5 * rescaled_x ** 2)
    # print(gaussian_kernel.shape)
    # rescale
    # print("H!!!!")
    # print(gaussian_kernel)
    # print(n_samples, bandwidth)
    K = jnp.sum(gaussian_kernel, axis=1) / n_samples / bandwidth
    # print(K.shape)
    # print("Byeeee")
    return K


def gaussian_kde_pdf(eval_points, samples, bandwidth):

    # distances (use broadcasting)
    rescaled_x = (eval_points - samples) / bandwidth

    # compute the gaussian kernel
    gaussian_kernel = jnp.exp(-0.5 * rescaled_x ** 2) / jnp.sqrt(2 * jnp.pi)

    # rescale
    return jnp.sum(gaussian_kernel, axis=0) / samples.shape[0] / bandwidth


def normalization_factor(data, bw):

    data_covariance = jnp.cov(data[:, jnp.newaxis], rowvar=0, bias=False)

    covariance = data_covariance * bw ** 2

    stdev = jnp.sqrt(covariance)

    return stdev


def gaussian_kde_cdf(x_eval, samples, factor):

    low = jnp.ravel((-jnp.inf - samples) / factor)
    hi = jnp.ravel((x_eval - samples) / factor)

    return jax.scipy.special.ndtr(hi - low).mean(axis=0)


def broadcast_kde_cdf(x_evals, samples, factor):
    return jax.scipy.special.ndtr(
        (x_evals[:, jnp.newaxis] - samples[jnp.newaxis, :]) / factor
    ).mean(axis=1)


def estimate_bw(n_samples, n_features, method="scott"):
    if isinstance(method, float) or isinstance(method, Array):
        return method
    elif method == "scott":
        return scotts_method(n_samples, n_features,)
    elif method == "silverman":
        return silvermans_method(n_samples, n_features,)
    else:
        raise ValueError(f"Unrecognized bw estimation method: {method}")


def scotts_method(
    n_samples: int, n_features: int,
):
    return jnp.power(n_samples, -1.0 / (n_features + 4))


def silvermans_method(
    n_samples: int, n_features: int,
):
    return jnp.power(n_samples * (n_features + 2.0) / 4.0, -1.0 / (n_features + 4))


def kde_forward_transform(params: UniKDEParams, X: Array) -> Array:
    """Forward univariate uniformize transformation
    
    Parameters
    ----------
    X : jnp.ndarray
        The univariate data to be transformed.
    
    params: UniParams
        the tuple containing the params. 
        See `rbig_jax.transforms.uniformize` for details.
    
    Returns
    -------
    X_trans : jnp.ndarray
        The transformed univariate parameters
    """
    return jnp.interp(X, params.support, params.quantiles)


def kde_inverse_transform(params: UniKDEParams, X: Array) -> Array:
    """Inverse univariate uniformize transformation
    
    Parameters
    ----------
    X : jnp.ndarray
        The uniform univariate data to be transformed.
    
    params: UniParams
        the tuple containing the params. 
        See `rbig_jax.transforms.histogram` for details.
    
    Returns
    -------
    X_trans : jnp.ndarray
        The transformed univariate parameters
    """
    return jnp.interp(X, params.quantiles, params.support)


def kde_gradient_transform(params: UniKDEParams, X: Array) -> Array:
    """Forward univariate uniformize transformation gradient
    
    Parameters
    ----------
    X : jnp.ndarray
        The univariate data to be transformed.
    
    params: UniParams
        the tuple containing the params. 
        See `rbig_jax.transforms.histogram` for details.
    
    Returns
    -------
    X_trans : jnp.ndarray
        The transformed univariate parameters
    """
    return jnp.interp(X, params.support_pdf, params.empirical_pdf)
