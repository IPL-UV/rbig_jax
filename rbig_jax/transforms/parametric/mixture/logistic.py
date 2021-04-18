from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from chex import Array, dataclass
from jax.nn import log_sigmoid, log_softmax, softplus
from jax.random import PRNGKey
from jax.scipy.special import logsumexp

from rbig_jax.transforms.base import Bijector
from rbig_jax.utils import bisection_search


@dataclass
class MixtureLogisticCDF(Bijector):
    means: Array
    log_scales: Array
    prior_logits: Array

    def forward_and_log_det(self, inputs: Array, **kwargs) -> Tuple[Array, Array]:

        # # forward transformation with batch dimension
        # outputs = mixture_logistic_cdf_vectorized(
        #     inputs, self.prior_logits, self.means, jnp.exp(self.log_scales),
        # )

        outputs = mixture_logistic_cdf(
            inputs, self.prior_logits, self.means, jnp.exp(self.log_scales),
        )

        # log abs det, all zeros
        logabsdet = mixture_logistic_log_pdf_vectorized(
            inputs, self.prior_logits, self.means, jnp.exp(self.log_scales),
        )

        return outputs, logabsdet  # .sum(axis=1)

    def inverse_and_log_det(self, inputs: Array, **kwargs) -> Tuple[Array, Array]:

        # transformation
        outputs = mixture_logistic_invcdf_vectorized(
            inputs, self.prior_logits, self.means, jnp.exp(self.log_scales),
        )
        # log abs det, all zeros
        logabsdet = mixture_logistic_log_pdf_vectorized(
            outputs, self.prior_logits, self.means, jnp.exp(self.log_scales),
        )

        return outputs, logabsdet  # .sum(axis=1)


def InitMixtureLogisticCDF(n_components: int) -> Callable:
    """Performs the householder transformation.

    This is a useful method to parameterize an orthogonal matrix.
    
    Parameters
    ----------
    n_features : int
        the number of features of the data
    n_reflections: int
        the number of householder reflections
    """

    def init_func(rng: PRNGKey, n_features: int, **kwargs) -> MixtureLogisticCDF:

        # initialize mixture
        means = jax.random.normal(key=rng, shape=(n_features, n_components))
        log_scales = jnp.zeros((n_features, n_components))
        prior_logits = jnp.zeros((n_features, n_components))

        init_params = MixtureLogisticCDF(
            means=means, log_scales=log_scales, prior_logits=prior_logits
        )

        return init_params

    return init_func


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
    x_r = jnp.expand_dims(x, axis=-1)
    #
    # x_r =jnp.tile(x, (n_features, n_components))
    # print(x.shape, x_r.shape)
    # normalize logit weights to 1, (D,K)->(D,K)
    prior_logits = log_softmax(prior_logits, axis=-1)

    # calculate the log pdf, (D,K)->(D,K)
    log_cdfs = prior_logits + logistic_log_cdf(x_r, means, scales)

    # normalize distribution for components, (D,K)->(D,)
    log_cdf = logsumexp(log_cdfs, axis=-1)

    return jnp.exp(log_cdf)


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
    init_lb = jnp.ones_like(means).max(axis=1) - 1_000.0
    init_ub = jnp.ones_like(means).max(axis=1) + 1_000.0

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
    x_r = jnp.expand_dims(x, axis=-1)

    # normalize logit weights to 1, (D,K)->(D,K)
    prior_logits = log_softmax(prior_logits, axis=-1)

    # calculate the log pdf, (D,K)->(D,K)
    # print(x.shape, prior_logits.shape, )
    log_pdfs = prior_logits + logistic_log_pdf(x_r, means, scales)
    # print("Log PDFS:", log_pdfs.shape)

    # normalize distribution for components, (D,K)->(D,)
    log_pdf = logsumexp(log_pdfs, axis=-1)

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
    # log_prob = z -jnp.log(scale) - 2 * jax.nn.softplus(z)
    # log_prob = jax.scipy.stats.logistic.logpdf(z)
    log_prob = z - jnp.log(scale) - 2 * softplus(z)

    return log_prob
