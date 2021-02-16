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

        # get transformation
        z = mixture_gaussian_cdf(
            x, self.prior_logits.value, self.means.value, np.exp(self.log_scales.value)
        )

        # get log_determinant jacobian
        log_abs_det = mixture_gaussian_log_pdf(
            x, self.prior_logits.value, self.means.value, np.exp(self.log_scales.value)
        )

        return z, log_abs_det

    def transform(self, x: JaxArray) -> Tuple[JaxArray, JaxArray]:

        # get transformation
        z = mixture_gaussian_cdf(
            x, self.prior_logits.value, self.means.value, np.exp(self.log_scales.value)
        )

        return z

    def inverse(self, z: JaxArray) -> JaxArray:
        # INITIALIZE BOUNDS
        init_lb = np.ones_like(self.means.value).max(axis=1) - 1_000.0
        init_ub = np.ones_like(self.means.value).max(axis=1) + 1_000.0

        # INITIALIZE FUNCTION
        f = jax.partial(
            mixture_gaussian_cdf,
            prior_logits=self.prior_logits.value,
            means=self.means.value,
            scales=np.exp(self.log_scales.value),
        )

        return bisection_search(f, z, init_lb, init_ub)


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
        log_cdf (JaxArray) : log CDF for the mixture distribution
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
