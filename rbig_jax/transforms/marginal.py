from typing import Callable, List, NamedTuple, Tuple
import jax
import jax.numpy as np

from rbig_jax.transforms.uniformize import UniParams


def marginal_transform_params(X, function: Callable):

    X, params = jax.vmap(function, out_axes=(0, 1))(X.T)
    return (
        np.stack(X, axis=1),
        UniParams(
            support=params.support.T,
            quantiles=params.quantiles.T,
            support_pdf=params.support_pdf.T,
            empirical_pdf=params.empirical_pdf.T,
        ),
    )


def marginal_transform(X, function: Callable, params: List[NamedTuple]) -> np.ndarray:

    X = jax.vmap(function, in_axes=(0, 0), out_axes=0)(X.T, params)

    return np.vstack(X).T


def marginal_transform_gradient(
    X, function: Callable, params: List[NamedTuple]
) -> np.ndarray:

    X, log_abs_det = jax.vmap(function, in_axes=(0, 0), out_axes=(0, 0))(X.T, params)

    return np.vstack(X).T, np.vstack(log_abs_det).T


# def forward_uniformization(X, params):
#     return (
#         np.interp(X, params.support, params.quantiles),
#         np.log(np.interp(X, params.support_pdf, params.empirical_pdf)),
#     )


# def inverse_uniformization(X, params):
#     return np.interp(X, params.quantiles, params.support)


# def forward_inversecdf(X):
#     return jax.scipy.stats.norm.ppf(X)


# def inverse_inversecdf(X):
#     return jax.scipy.stats.norm.cdf(X)


# def forward_gaussianization(X, params):

#     # transform to uniform domain
#     X, Xdj = forward_uniformization(X, params)

#     # clip boundaries
#     X = np.clip(X, 1e-5, 1.0 - 1e-5)

#     # transform to the gaussian domain
#     X = forward_inversecdf(X)

#     log_prob = Xdj - jax.scipy.stats.norm.logpdf(X)

#     return X, log_prob


# def inverse_gaussianization(X, params):

#     # transform to uniform domain
#     X = inverse_inversecdf(X)

#     # transform to the original domain
#     X = inverse_uniformization(X, params)

#     return X
