from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from chex import Array, dataclass
from jax.nn import log_softmax, softplus
from jax.random import PRNGKey
from jax.scipy.special import logsumexp

from rbig_jax.transforms.base import Bijector, InitLayersFunctions
from rbig_jax.transforms.parametric.mixture.init import init_mixture_weights
from rbig_jax.utils import bisection_search


@dataclass
class MixtureGaussianCDF(Bijector):
    means: Array
    log_scales: Array
    prior_logits: Array

    def forward_and_log_det(self, inputs: Array, **kwargs) -> Tuple[Array, Array]:

        # forward transformation with batch dimension
        outputs = mixture_gaussian_cdf(
            inputs, self.prior_logits, self.means, softplus(self.log_scales),
        )

        # log abs det, all zeros
        logabsdet = mixture_gaussian_log_pdf(
            inputs, self.prior_logits, self.means, softplus(self.log_scales),
        )

        return outputs, logabsdet  # .sum(axis=1)

    def inverse_and_log_det(self, inputs: Array, **kwargs) -> Tuple[Array, Array]:

        # transformation
        outputs = mixture_gaussian_invcdf_vectorized(
            inputs, self.prior_logits, self.means, softplus(self.log_scales),
        )
        # log abs det, all zeros
        logabsdet = mixture_gaussian_log_pdf(
            outputs, self.prior_logits, self.means, softplus(self.log_scales),
        )

        return outputs, logabsdet  # .sum(axis=1)


def InitMixtureGaussianCDF(
    n_components: int, init_method: str = "standard", seed: int = 123
) -> Callable:
    """Performs the householder transformation.

    This is a useful method to parameterize an orthogonal matrix.
    
    Parameters
    ----------
    n_features : int
        the number of features of the data
    n_reflections: int
        the number of householder reflections
    """

    def bijector(
        inputs, n_features: int, rng: PRNGKey = None, **kwargs
    ) -> MixtureGaussianCDF:
        prior_logits, means, log_scales = init_mixture_weights(
            rng=seed if rng is None else rng,
            n_features=n_features,
            n_components=n_components,
            method=init_method,
            X=inputs,
        )

        bijector = MixtureGaussianCDF(
            means=means, log_scales=log_scales, prior_logits=prior_logits
        )

        return bijector

    def transform_and_bijector(
        inputs, n_features: int, rng: PRNGKey = None, **kwargs
    ) -> MixtureGaussianCDF:
        prior_logits, means, log_scales = init_mixture_weights(
            rng=seed if rng is None else rng,
            n_features=n_features,
            n_components=n_components,
            method=init_method,
            X=inputs,
        )

        bijector = MixtureGaussianCDF(
            means=means, log_scales=log_scales, prior_logits=prior_logits
        )
        # forward transform
        outputs = bijector.forward(inputs=inputs)
        return outputs, bijector

    def transform(
        inputs, n_features: int, rng: PRNGKey = None, **kwargs
    ) -> MixtureGaussianCDF:

        prior_logits, means, log_scales = init_mixture_weights(
            rng=seed if rng is None else rng,
            n_features=n_features,
            n_components=n_components,
            method=init_method,
            X=inputs,
        )

        bijector = MixtureGaussianCDF(
            means=means, log_scales=log_scales, prior_logits=prior_logits
        )
        # forward transform
        outputs = bijector.forward(inputs=inputs)

        return outputs

    def transform_gradient_bijector(
        inputs, n_features: int, rng: PRNGKey = None, **kwargs
    ) -> MixtureGaussianCDF:
        prior_logits, means, log_scales = init_mixture_weights(
            rng=seed if rng is None else rng,
            n_features=n_features,
            n_components=n_components,
            method=init_method,
            X=inputs,
        )

        bijector = MixtureGaussianCDF(
            means=means, log_scales=log_scales, prior_logits=prior_logits
        )
        # forward transform
        outputs, logabsdet = bijector.forward_and_log_Det(inputs=inputs)
        return outputs, logabsdet, bijector

    return InitLayersFunctions(
        transform=transform,
        bijector=bijector,
        transform_and_bijector=transform_and_bijector,
        transform_gradient_bijector=transform_gradient_bijector,
    )


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

    x = jnp.expand_dims(x, axis=-1)

    # normalize
    x = (x - means) / scales

    # normalize prior logits
    prior_logits = jax.nn.log_softmax(prior_logits, axis=-1)

    log_cdfs = prior_logits + jax.scipy.stats.norm.logcdf(
        x
    )  # jax.scipy.special.log_ndtr(x)

    # calculate log cdf
    log_cdfs = jax.nn.logsumexp(log_cdfs, axis=-1)

    return jnp.exp(log_cdfs)


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
    # x_r = x.reshape(-1, 1)
    x = jnp.expand_dims(x, axis=-1)

    x = (x - means) / scales
    # normalize logit weights to 1, (D,K)->(D,K)
    prior_logits = log_softmax(prior_logits, axis=-1)

    # calculate the log pdf, (D,K)->(D,K)
    log_pdfs = prior_logits + jax.scipy.stats.norm.logpdf(x)

    # normalize distribution for components, (D,K)->(D,)
    log_pdf = logsumexp(log_pdfs, axis=-1)

    return log_pdf


mixture_gaussian_log_pdf_vectorized = jax.vmap(
    mixture_gaussian_log_pdf, in_axes=(0, None, None, None)
)
