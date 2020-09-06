import jax
import jax.numpy as np


def forward_uniformization(X, params):
    return (
        np.interp(X, params.support, params.quantiles),
        np.log(np.interp(X, params.support_pdf, params.empirical_pdf)),
    )


def inverse_uniformization(X, params):
    return np.interp(X, params.quantiles, params.support)


def forward_inversecdf(X):
    return jax.scipy.stats.norm.ppf(X)


def inverse_inversecdf(X):
    return jax.scipy.stats.norm.cdf(X)


def forward_gaussianization(X, params):

    # transform to uniform domain
    X, Xdj = forward_uniformization(X, params)

    # clip boundaries
    X = np.clip(X, 1e-5, 1.0 - 1e-5)

    # transform to the gaussian domain
    X = forward_inversecdf(X)

    log_prob = Xdj - jax.scipy.stats.norm.logpdf(X)

    return X, log_prob


def inverse_gaussianization(X, params):

    # transform to uniform domain
    X = inverse_inversecdf(X)

    # transform to the original domain
    X = inverse_uniformization(X, params)

    return X
