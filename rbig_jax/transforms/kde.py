from functools import partial
from typing import Union

import jax
import jax.numpy as np
from chex import Array, dataclass
from rbig_jax.utils import get_domain_extension


@dataclass
class UniKDEParams:
    support: Array
    quantiles: Array
    support_pdf: Array
    empirical_pdf: Array


def InitKDEUniformize(
    n_samples: int,
    support_extension: Union[int, float] = 10,
    precision: int = 1_000,
    bw: float = 0.1,
):

    # TODO a kde bin init function
    # bw = scotts_method(n_samples, 1) * 0.5

    def init_fun(inputs):

        outputs, params = get_kde_params(
            X=inputs,
            support_extension=support_extension,
            precision=precision,
            bw=bw,
            return_params=True,
        )

        return outputs, params

    def forward_transform(params, inputs):

        return np.interp(inputs, params.support, params.quantiles)

    def gradient_transform(params, inputs):
        return np.interp(inputs, params.support_pdf, params.empirical_pdf)

    def inverse_transform(params, inputs):
        return np.interp(inputs, params.quantiles, params.support)

    return init_fun, forward_transform, gradient_transform, inverse_transform


def get_kde_params(
    X: np.ndarray,
    bw: int = 0.1,
    support_extension: Union[int, float] = 10,
    precision: int = 1_000,
    return_params: bool = True,
):
    # generate support points
    lb, ub = get_domain_extension(X, support_extension)
    support = np.linspace(lb, ub, precision)

    # calculate the pdf for gaussian pdf
    pdf_support = broadcast_kde_pdf(support, X, bw)

    # calculate the cdf for support points
    factor = normalization_factor(X, bw)

    quantiles = broadcast_kde_cdf(support, X, factor)

    # forward transformation
    outputs = np.interp(X, support, quantiles)

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
    X: np.ndarray, support_extension: Union[int, float] = 10, precision: int = 1_000,
):
    # generate support points
    lb, ub = get_domain_extension(X, support_extension)
    grid = np.linspace(lb, ub, precision)

    bw = scotts_method(X.shape[0], 1) * 0.5

    # calculate the cdf for grid points
    factor = normalization_factor(X, bw)

    x_cdf = broadcast_kde_cdf(grid, X, factor)

    X_transform = np.interp(X, grid, x_cdf)

    return X_transform


def broadcast_kde_pdf(eval_points, samples, bandwidth):

    n_samples = samples.shape[0]
    # print(n_samples, bandwidth)

    # distances (use broadcasting)
    rescaled_x = (
        eval_points[:, np.newaxis] - samples[np.newaxis, :]
    ) / bandwidth  # (2 * bandwidth ** 2)
    # print(rescaled_x.shape)
    # compute the gaussian kernel
    gaussian_kernel = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * rescaled_x ** 2)
    # print(gaussian_kernel.shape)
    # rescale
    # print("H!!!!")
    # print(gaussian_kernel)
    # print(n_samples, bandwidth)
    K = np.sum(gaussian_kernel, axis=1) / n_samples / bandwidth
    # print(K.shape)
    # print("Byeeee")
    return K


def gaussian_kde_pdf(eval_points, samples, bandwidth):

    # distances (use broadcasting)
    rescaled_x = (eval_points - samples) / bandwidth

    # compute the gaussian kernel
    gaussian_kernel = np.exp(-0.5 * rescaled_x ** 2) / np.sqrt(2 * np.pi)

    # rescale
    return np.sum(gaussian_kernel, axis=0) / samples.shape[0] / bandwidth


def normalization_factor(data, bw):

    data_covariance = np.cov(data[:, np.newaxis], rowvar=0, bias=False)

    covariance = data_covariance * bw ** 2

    stdev = np.sqrt(covariance)

    return stdev


def gaussian_kde_cdf(x_eval, samples, factor):

    low = np.ravel((-np.inf - samples) / factor)
    hi = np.ravel((x_eval - samples) / factor)

    return jax.scipy.special.ndtr(hi - low).mean(axis=0)


def broadcast_kde_cdf(x_evals, samples, factor):
    return jax.scipy.special.ndtr(
        (x_evals[:, np.newaxis] - samples[np.newaxis, :]) / factor
    ).mean(axis=1)


def scotts_method(n, d=1):
    return np.power(n, -1.0 / (d + 4))


def silvermans_method(n, d=1):
    return np.power(n * (d + 2.0) / 4.0, -1.0 / (d + 4))
