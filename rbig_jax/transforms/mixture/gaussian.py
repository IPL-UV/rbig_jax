from typing import Tuple

import jax
import jax.numpy as np
import objax
from jax.nn import log_softmax, softplus
from jax.scipy.special import logsumexp
from objax import TrainRef, TrainVar
from objax.typing import JaxArray

from rbig_jax.transforms.base import Transform
from rbig_jax.utils import bisection_search


class MixtureGaussianCDF(Transform):
    def __init__(self, n_features: int, n_components: int) -> None:

        # initialize variables
        self.means = TrainVar(objax.random.normal((n_features, n_components)))
        self.log_scales = TrainVar(np.zeros((n_features, n_components)))
        self.prior_logits = TrainVar(np.zeros((n_features, n_components)))

    def __call__(self, x: JaxArray) -> Tuple[JaxArray, JaxArray]:

        # transformation
        z = mixture_gaussian_cdf_vectorized(
            x, self.prior_logits.value, self.means.value, np.exp(self.log_scales.value),
        )

        # get log_determinant jacobian
        log_abs_det = mixture_gaussian_log_pdf_vectorized(
            x, self.prior_logits.value, self.means.value, np.exp(self.log_scales.value),
        )

        return z, log_abs_det.sum(axis=1)

    def transform(self, x: JaxArray) -> Tuple[JaxArray, JaxArray]:

        # transformation
        z = mixture_gaussian_cdf_vectorized(
            x, self.prior_logits.value, self.means.value, np.exp(self.log_scales.value),
        )

        return z

    def inverse(self, z: JaxArray) -> JaxArray:
        # transformation
        z = mixture_gaussian_invcdf_vectorized(
            z, self.prior_logits.value, self.means.value, np.exp(self.log_scales.value),
        )

        return z


def mixture_gaussian_cdf(
    x: JaxArray, prior_logits: JaxArray, means: JaxArray, scales: JaxArray
) -> JaxArray:
    """
    Args:
        x (JaxArray): input vector
            (D,)
        prior_logits (JaxArray): prior logits to weight the components
            (D, K)
        means (JaxArray): means per component per feature
            (D, K)
        scales (JaxArray): scales per component per feature
            (D, K)
    Returns:
        x_cdf (JaxArray) : CDF for the mixture distribution
    """
    # n_features, n_components = prior_logits
    #
    # x_r = np.tile(x, (n_features, n_components))
    x_r = x.reshape(-1, 1)
    # normalize logit weights to 1
    prior_logits = jax.nn.log_softmax(prior_logits)

    # calculate the log cdf
    log_cdfs = prior_logits + jax.scipy.stats.norm.logcdf(x_r, means, scales)

    # normalize distribution
    log_cdf = jax.scipy.special.logsumexp(log_cdfs, axis=1)

    return np.exp(log_cdf)


mixture_gaussian_cdf_vectorized = jax.vmap(
    mixture_gaussian_cdf, in_axes=(0, None, None, None)
)


def mixture_gaussian_invcdf(
    x: JaxArray, prior_logits: JaxArray, means: JaxArray, scales: JaxArray
) -> JaxArray:
    """
    Args:
        x (JaxArray): input vector
            (D,)
        prior_logits (JaxArray): prior logits to weight the components
            (D, K)
        means (JaxArray): means per component per feature
            (D, K)
        scales (JaxArray): scales per component per feature
            (D, K)
    Returns:
        x_invcdf (JaxArray) : log CDF for the mixture distribution
    """
    # INITIALIZE BOUNDS
    init_lb = np.ones_like(means).max(axis=1) - 1_000.0
    init_ub = np.ones_like(means).max(axis=1) + 1_000.0

    # INITIALIZE FUNCTION
    f = jax.partial(
        mixture_gaussian_cdf, prior_logits=prior_logits, means=means, scales=scales,
    )

    return bisection_search(f, x, init_lb, init_ub)


mixture_gaussian_invcdf_vectorized = jax.vmap(
    mixture_gaussian_invcdf, in_axes=(0, None, None, None)
)


def mixture_gaussian_log_pdf(
    x: JaxArray, prior_logits: JaxArray, means: JaxArray, scales: JaxArray
) -> JaxArray:
    """
    Args:
        x (JaxArray): input vector
            (D,)
        prior_logits (JaxArray): prior logits to weight the components
            (D, K)
        means (JaxArray): means per component per feature
            (D, K)
        scales (JaxArray): scales per component per feature
            (D, K)
    Returns:
        log_pdf (JaxArray) : log PDF for the mixture distribution
    """
    # n_components = prior_logits.shape[1]
    #
    # x_r = np.tile(x, (n_components))
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
