import collections
from functools import partial
from typing import Union

import jax
import jax.numpy as np

from rbig_jax.transforms.utils import get_domain_extension

Params = collections.namedtuple(
    "Params", ["support", "quantiles", "support_pdf", "empirical_pdf"]
)


def get_kde_params(
    X: np.ndarray, support_extension: Union[int, float] = 10, precision: int = 1_000,
):
    # generate support points
    lb, ub = get_domain_extension(X, support_extension)
    grid = np.linspace(lb, ub, precision)

    bw = scotts_method(X.shape[0], 1) * 0.5

    # calculate the pdf for gaussian pdf
    # print(grid.shape, X.shape, bw.shape)
    x_pdf = broadcast_kde_pdf(grid, X, bw)
    # print(x_pdf.shape)
    # print(x_pdf.shape)

    # calculate the cdf for grid points
    factor = normalization_factor(X, bw)
    # print("Before CDF:", grid.shape, X.shape, factor)
    x_cdf = broadcast_kde_cdf(grid, X, factor)
    # print("CDF:", x_pdf.shape)
    X_transform = np.interp(X, grid, x_cdf)
    # print(grid.shape, x_cdf.shape, x_pdf.shape, X.shape)

    X_ldj = np.log(np.interp(X, grid, x_pdf))
    # print(X_ldj.shape)
    return (
        X_transform,
        X_ldj,
        Params(grid, x_cdf, grid, x_pdf),
    )


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
