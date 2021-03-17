from typing import Tuple, Callable

import jax
import jax.numpy as np
from jax.nn import log_sigmoid, log_softmax, softplus
from jax.scipy.special import logsumexp
from chex import Array
from jax.random import PRNGKey
from rbig_jax.utils import bisection_search
from rbig_jax.transforms.parametric.mixture.gaussian import MixtureParams


def MixtureLogisticCDF(n_components: int) -> Callable:
    """Performs the householder transformation.

    This is a useful method to parameterize an orthogonal matrix.
    
    Parameters
    ----------
    n_features : int
        the number of features of the data
    n_reflections: int
        the number of householder reflections
    """

    def init_func(
        rng: PRNGKey, n_features: int, **kwargs
    ) -> Tuple[MixtureParams, Callable, Callable]:

        # initialize mixture
        means = jax.random.normal(key=rng, shape=(n_features, n_components))
        log_scales = np.zeros((n_features, n_components))
        prior_logits = np.zeros((n_features, n_components))

        init_params = MixtureParams(
            means=means, log_scales=log_scales, prior_logits=prior_logits
        )

        def forward_func(
            params: MixtureParams, inputs: Array, **kwargs
        ) -> Tuple[Array, Array]:

            # forward transformation with batch dimension
            outputs = mixture_logistic_cdf_vectorized(
                inputs, params.prior_logits, params.means, np.exp(params.log_scales),
            )

            # log abs det, all zeros
            logabsdet = mixture_logistic_log_pdf_vectorized(
                inputs, params.prior_logits, params.means, np.exp(params.log_scales),
            )

            return outputs, logabsdet.sum(axis=1)

        def inverse_func(
            params: MixtureParams, inputs: Array, **kwargs
        ) -> Tuple[Array, Array]:

            # transformation
            outputs = mixture_logistic_invcdf_vectorized(
                inputs, params.prior_logits, params.means, np.exp(params.log_scales),
            )
            # log abs det, all zeros
            logabsdet = mixture_logistic_log_pdf_vectorized(
                outputs, params.prior_logits, params.means, np.exp(params.log_scales),
            )

            return outputs, logabsdet.sum(axis=1)

        return init_params, forward_func, inverse_func

    return init_func


# class MixtureLogisticCDF(Transform):
#     def __init__(self, n_features: int, n_components: int) -> None:

#         # initialize variables
#         self.means = TrainVar(objax.random.normal((n_features, n_components)))
#         self.log_scales = TrainVar(np.zeros((n_features, n_components)))
#         self.prior_logits = TrainVar(np.zeros((n_features, n_components)))

#     def __call__(self, x: Array) -> Tuple[Array, Array]:

#         # transformation
#         z = mixture_logistic_cdf_vectorized(
#             x, self.prior_logits.value, self.means.value, np.exp(self.log_scales.value),
#         )

#         # get log_determinant jacobian
#         log_abs_det = mixture_logistic_log_pdf_vectorized(
#             x, self.prior_logits.value, self.means.value, np.exp(self.log_scales.value),
#         )

#         return z, log_abs_det.sum(axis=1)

#     def transform(self, x: Array) -> Tuple[Array, Array]:

#         # transformation
#         z = mixture_logistic_cdf_vectorized(
#             x, self.prior_logits.value, self.means.value, np.exp(self.log_scales.value),
#         )

#         return z

#     def inverse(self, z: Array) -> Array:
#         # transformation
#         z = mixture_logistic_invcdf_vectorized(
#             z, self.prior_logits.value, self.means.value, np.exp(self.log_scales.value),
#         )

#         return z


def mixture_logistic_cdf(
    x: Array, prior_logits: Array, means: Array, scales: Array
) -> Array:
    """
    Args:
        x (Array): input vector
            (D,)
        prior_logits (Array): prior logits to weight the components
            (D, K)
        means (Array): means per component per feature
            (D, K)
        scales (Array): scales per component per feature
            (D, K)
    Returns:
        log_cdf (Array) : log CDF for the mixture distribution
    """
    # print(prior_logits.shape)
    # n_features, n_components = prior_logits
    x_r = x.reshape(-1, 1)
    #
    # x_r = np.tile(x, (n_features, n_components))
    # print(x.shape, x_r.shape)
    # normalize logit weights to 1, (D,K)->(D,K)
    prior_logits = log_softmax(prior_logits, axis=1)

    # calculate the log pdf, (D,K)->(D,K)
    log_cdfs = prior_logits + logistic_log_cdf(x_r, means, scales)

    # normalize distribution for components, (D,K)->(D,)
    log_cdf = logsumexp(log_cdfs, axis=1)

    return np.exp(log_cdf)


mixture_logistic_cdf_vectorized = jax.vmap(
    mixture_logistic_cdf, in_axes=(0, None, None, None)
)


def logistic_log_cdf(x: Array, mean: Array, scale: Array) -> Array:
    """Element-wise log CDF of the logistic distribution

    Parameters
    ----------
    x : Array
        a feature vector to be transformed, shape=(n_features,)
    mean : Array
        mean components to be transformed, shape=(n_components,)
    scale : Array
        scale components to be transformed, shape=(n_components,)

    Returns
    -------
    log_cdf (Array): log cdf of the distribution
    """

    # change of variables
    z = (x - mean) / scale

    # log cdf
    log_cdf = log_sigmoid(z)

    return log_cdf


def mixture_logistic_invcdf(
    x: Array, prior_logits: Array, means: Array, scales: Array
) -> Array:
    """
    Args:
        x (Array): input vector
            (D,)
        prior_logits (Array): prior logits to weight the components
            (D, K)
        means (Array): means per component per feature
            (D, K)
        scales (Array): scales per component per feature
            (D, K)
    Returns:
        x_invcdf (Array) : log CDF for the mixture distribution
    """
    # INITIALIZE BOUNDS
    init_lb = np.ones_like(means).max(axis=1) - 1_000.0
    init_ub = np.ones_like(means).max(axis=1) + 1_000.0

    # INITIALIZE FUNCTION
    f = jax.partial(
        mixture_logistic_cdf, prior_logits=prior_logits, means=means, scales=scales,
    )

    return bisection_search(f, x, init_lb, init_ub)


mixture_logistic_invcdf_vectorized = jax.vmap(
    mixture_logistic_invcdf, in_axes=(0, None, None, None)
)


def mixture_logistic_log_pdf(
    x: Array, prior_logits: Array, means: Array, scales: Array
) -> Array:
    """
    Args:
        x (Array): input vector
            (D,)
        prior_logits (Array): prior logits to weight the components
            (D, K)
        means (Array): means per component per feature
            (D, K)
        scales (Array): scales per component per feature
            (D, K)
    Returns:
        log_pdf (Array) : log PDF for the mixture distribution
    """
    # n_components = prior_logits.shape[1]
    #

    # add component dimension, (D,)->(D,1)
    # will allow for broadcasting
    x_r = x.reshape(-1, 1)

    # normalize logit weights to 1, (D,K)->(D,K)
    prior_logits = log_softmax(prior_logits, axis=1)

    # calculate the log pdf, (D,K)->(D,K)
    # print(x.shape, prior_logits.shape, )
    log_pdfs = prior_logits + logistic_log_pdf(x_r, means, scales)
    # print("Log PDFS:", log_pdfs.shape)

    # normalize distribution for components, (D,K)->(D,)
    log_pdf = logsumexp(log_pdfs, axis=1)

    return log_pdf


mixture_logistic_log_pdf_vectorized = jax.vmap(
    mixture_logistic_log_pdf, in_axes=(0, None, None, None)
)


def logistic_log_pdf(x: Array, mean: Array, scale: Array) -> Array:
    """Element-wise log PDF of the logistic distribution

    Args:
        x (Array): feature vector to be transformed
        mean (Array) : mean for the features
        scale (Array) : scale for features

    Returns:
        log_prob (Array): log probability of the distribution
    """

    # change of variables
    z = (x - mean) / scale

    # log probability
    # log_prob = z - np.log(scale) - 2 * jax.nn.softplus(z)
    # log_prob = jax.scipy.stats.logistic.logpdf(z)
    log_prob = z - np.log(scale) - 2 * softplus(z)

    return log_prob

