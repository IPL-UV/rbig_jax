from typing import Tuple
import jax
import objax
from objax import TrainVar, TrainRef
from objax.typing import JaxArray
import jax.numpy as np
from jax.scipy.special import logsumexp
from jax.nn import log_softmax, softplus, log_sigmoid
from rbig_jax.utils import bisection_search


class MixtureLogisticCDF(objax.Module):
    def __init__(self, n_features: int, n_components: int) -> None:

        # initialize variables
        self.means = TrainVar(objax.random.normal((n_features, n_components)))
        self.log_scales = TrainVar(np.zeros((n_features, n_components)))
        self.prior_logits = TrainVar(np.zeros((n_features, n_components)))

    def __call__(self, x: JaxArray) -> Tuple[JaxArray, JaxArray]:

        # get transformation
        z = mixture_logistic_cdf(
            x, self.prior_logits.value, self.means.value, np.exp(self.log_scales.value)
        )
        # get log_determinant jacobian
        log_abs_det = mixture_logistic_log_pdf(
            x, self.prior_logits.value, self.means.value, np.exp(self.log_scales.value)
        )

        return z, log_abs_det

    def inverse(self, z: JaxArray) -> JaxArray:
        # INITIALIZE BOUNDS
        init_lb = np.ones_like(self.means.value).max(axis=1) - 1_000.0
        init_ub = np.ones_like(self.means.value).max(axis=1) + 1_000.0

        # INITIALIZE FUNCTION
        f = jax.partial(
            mixture_logistic_cdf,
            prior_logits=self.prior_logits.value,
            means=self.means.value,
            scales=np.exp(self.log_scales.value),
        )

        return bisection_search(f, z, init_lb, init_ub)


def mixture_logistic_log_pdf(
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


def mixture_logistic_cdf(
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


def logistic_log_cdf(x: JaxArray, mean: JaxArray, scale: JaxArray) -> JaxArray:
    """Element-wise log CDF of the logistic distribution

    Args:
        x (JaxArray): feature vector to be transformed
        mean (JaxArray) : mean for the features
        scale (JaxArray) : scale for features

    Returns:
        log_cdf (JaxArray): log probability of the distribution
    """

    # change of variables
    z = (x - mean) / scale

    # log cdf
    # log_cdf = np.log(jax.scipy.stats.logistic.cdf(z))
    log_cdf = log_sigmoid(z)

    return log_cdf

