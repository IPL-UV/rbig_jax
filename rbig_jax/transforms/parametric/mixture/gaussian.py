from typing import Tuple, Callable

import jax
import jax.numpy as jnp
from jax.nn import log_softmax
from jax.scipy.special import logsumexp
from chex import Array, dataclass
from jax.random import PRNGKey
from rbig_jax.utils import bisection_search


@dataclass
class MixtureParams:
    means: Array
    log_scales: Array
    prior_logits: Array


def MixtureGaussianCDF(n_components: int) -> Callable:
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
        log_scales = jnp.zeros((n_features, n_components))
        prior_logits = jnp.zeros((n_features, n_components))

        init_params = MixtureParams(
            means=means, log_scales=log_scales, prior_logits=prior_logits
        )

        def forward_func(
            params: MixtureParams, inputs: Array, **kwargs
        ) -> Tuple[Array, Array]:

            # forward transformation with batch dimension
            outputs = mixture_gaussian_cdf_vectorized(
                inputs, params.prior_logits, params.means, jnp.exp(params.log_scales),
            )

            # log abs det, all zeros
            logabsdet = mixture_gaussian_log_pdf_vectorized(
                inputs, params.prior_logits, params.means, jnp.exp(params.log_scales),
            )

            return outputs, logabsdet.sum(axis=1)

        def inverse_func(
            params: MixtureParams, inputs: Array, **kwargs
        ) -> Tuple[Array, Array]:

            # transformation
            outputs = mixture_gaussian_invcdf_vectorized(
                inputs, params.prior_logits, params.means, jnp.exp(params.log_scales),
            )
            # log abs det, all zeros
            logabsdet = mixture_gaussian_log_pdf_vectorized(
                outputs, params.prior_logits, params.means, jnp.exp(params.log_scales),
            )

            return outputs, logabsdet.sum(axis=1)

        return init_params, forward_func, inverse_func

    return init_func


# class MixtureGaussianCDF(Transform):
#     def __init__(self, n_features: int, n_components: int) -> None:

#         # initialize variables
#         self.means = TrainVar(objax.random.normal((n_features, n_components)))
#         self.log_scales = TrainVar(jnp.zeros((n_features, n_components)))
#         self.prior_logits = TrainVar(jnp.zeros((n_features, n_components)))

#     def __call__(self, x: Array) -> Tuple[Array, Array]:

#         # transformation
#         z = mixture_gaussian_cdf_vectorized(
#             x, self.prior_logits.value, self.means.value, jnp.exp(self.log_scales.value),
#         )

#         # get log_determinant jacobian
#         log_abs_det = mixture_gaussian_log_pdf_vectorized(
#             x, self.prior_logits.value, self.means.value, jnp.exp(self.log_scales.value),
#         )

#         return z, log_abs_det.sum(axis=1)

#     def transform(self, x: Array) -> Tuple[Array, Array]:

#         # transformation
#         z = mixture_gaussian_cdf_vectorized(
#             x, self.prior_logits.value, self.means.value, jnp.exp(self.log_scales.value),
#         )

#         return z

#     def inverse(self, z: Array) -> Array:
#         # transformation
#         z = mixture_gaussian_invcdf_vectorized(
#             z, self.prior_logits.value, self.means.value, jnp.exp(self.log_scales.value),
#         )

#         return z


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
