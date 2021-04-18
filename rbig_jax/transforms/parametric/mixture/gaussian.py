from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from chex import Array, dataclass
from jax.nn import log_softmax
from jax.random import PRNGKey
from jax.scipy.special import logsumexp

from rbig_jax.transforms.base import Bijector
from rbig_jax.utils import bisection_search


@dataclass
class MixtureGaussianCDF(Bijector):
    means: Array
    log_scales: Array
    prior_logits: Array

    def forward_and_log_det(self, inputs: Array, **kwargs) -> Tuple[Array, Array]:

        # forward transformation with batch dimension
        outputs = mixture_gaussian_cdf_vectorized(
            inputs, self.prior_logits, self.means, jnp.exp(self.log_scales),
        )

        # log abs det, all zeros
        logabsdet = mixture_gaussian_log_pdf_vectorized(
            inputs, self.prior_logits, self.means, jnp.exp(self.log_scales),
        )

        return outputs, logabsdet  # .sum(axis=1)

    def inverse_and_log_det(self, inputs: Array, **kwargs) -> Tuple[Array, Array]:

        # transformation
        outputs = mixture_gaussian_invcdf_vectorized(
            inputs, self.prior_logits, self.means, jnp.exp(self.log_scales),
        )
        # log abs det, all zeros
        logabsdet = mixture_gaussian_log_pdf_vectorized(
            outputs, self.prior_logits, self.means, jnp.exp(self.log_scales),
        )

        return outputs, logabsdet  # .sum(axis=1)


def InitMixtureGaussianCDF(n_components: int) -> Callable:
    """Performs the householder transformation.

    This is a useful method to parameterize an orthogonal matrix.
    
    Parameters
    ----------
    n_features : int
        the number of features of the data
    n_reflections: int
        the number of householder reflections
    """

    def init_func(rng: PRNGKey, n_features: int, **kwargs) -> MixtureGaussianCDF:

        # initialize mixture
        means = jax.random.normal(key=rng, shape=(n_features, n_components))
        log_scales = jnp.zeros((n_features, n_components))
        prior_logits = jnp.zeros((n_features, n_components))

        return MixtureGaussianCDF(
            means=means, log_scales=log_scales, prior_logits=prior_logits
        )

    return init_func


def mixture_gaussian_cdf(
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
        x_cdf (Array) : CDF for the mixture distribution
    """
    # n_features, n_components = prior_logits
    #
    # x_r = jnp.tile(x, (n_features, n_components))
    x_r = x.reshape(-1, 1)
    # normalize logit weights to 1
    prior_logits = jax.nn.log_softmax(prior_logits)

    # calculate the log cdf
    log_cdfs = prior_logits + jax.scipy.stats.norm.logcdf(x_r, means, scales)

    # normalize distribution
    log_cdf = jax.scipy.special.logsumexp(log_cdfs, axis=1)

    return jnp.exp(log_cdf)


mixture_gaussian_cdf_vectorized = jax.vmap(
    mixture_gaussian_cdf, in_axes=(0, None, None, None)
)


def mixture_gaussian_invcdf(
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
        mixture_gaussian_cdf, prior_logits=prior_logits, means=means, scales=scales,
    )

    return bisection_search(f, x, init_lb, init_ub)


mixture_gaussian_invcdf_vectorized = jax.vmap(
    mixture_gaussian_invcdf, in_axes=(0, None, None, None)
)


def mixture_gaussian_log_pdf(
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
    # x_r = jnp.tile(x, (n_components))
    x_r = x.reshape(-1, 1)
    # normalize logit weights to 1, (D,K)->(D,K)
    prior_logits = log_softmax(prior_logits, axis=1)

    # calculate the log pdf, (D,K)->(D,K)
    log_pdfs = prior_logits + jax.scipy.stats.norm.logpdf(x_r, means, scales)

    # normalize distribution for components, (D,K)->(D,)
    log_pdf = logsumexp(log_pdfs, axis=1)

    return log_pdf


mixture_gaussian_log_pdf_vectorized = jax.vmap(
    mixture_gaussian_log_pdf, in_axes=(0, None, None, None)
)
