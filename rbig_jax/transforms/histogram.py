import collections
from typing import Optional, Union, Tuple

import jax
import jax.numpy as np
import jax.numpy as jnp
from chex import Array, dataclass
from objax.typing import JaxArray
from distrax._src.bijectors.bijector import Bijector as distaxBijector
from rbig_jax.utils import get_domain_extension


class UniHistogramTransform(distaxBijector):
    def __init__(
        self, support: Array, quantiles: Array, support_pdf: Array, empirical_pdf: Array
    ):
        self.support = support
        self.quantiles = quantiles
        self.support_pdf = support_pdf
        self.empirical_pdf = empirical_pdf

    def forward(self, inputs: Array) -> Array:

        # transformation
        outputs = jnp.interp(inputs, self.support, self.quantiles)

        return outputs

    def inverse(self, inputs: Array) -> Array:

        # transformation
        outputs = jnp.interp(inputs, self.quantiles, self.support)

        return outputs

    def forward_log_det_jacobian(self, inputs: Array) -> Array:

        # gradient transformation
        abs_det = jnp.interp(inputs, self.support_pdf, self.empirical_pdf)

        log_abs_det = jnp.log(abs_det)

        return log_abs_det

    def inverse_log_det_jacobian(self, inputs: Array) -> Array:

        # gradient transformation
        abs_det = jnp.interp(inputs, self.support_pdf, self.empirical_pdf)

        log_abs_det = jnp.log(abs_det)

        return log_abs_det

    def forward_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:

        # transformation
        outputs = jnp.interp(inputs, self.support, self.quantiles)

        # gradient transformation
        abs_det = jnp.interp(inputs, self.support_pdf, self.empirical_pdf)

        log_abs_det = jnp.log(abs_det)

        return outputs, log_abs_det

    def inverse_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:

        outputs = jnp.interp(inputs, self.quantiles, self.support)

        # gradient transformation
        log_det = jnp.interp(inputs, self.support_pdf, self.empirical_pdf)

        log_abs_det = jnp.log(log_det)

        return outputs, log_abs_det


def InitUniHistTransform(
    n_samples: int,
    nbins: Optional[int] = None,
    support_extension: Union[int, float] = 10,
    precision: int = 1_000,
    alpha: float = 1e-5,
):

    # TODO a bin initialization function
    if nbins is None:
        nbins = int(np.sqrt(n_samples))

    def init_fun(inputs):

        outputs, bijector = init_hist_params(
            X=inputs,
            nbins=nbins,
            support_extension=support_extension,
            precision=precision,
            alpha=alpha,
            return_params=True,
        )

        return outputs, bijector

    return init_fun


def init_hist_params(
    X: jnp.ndarray,
    nbins: int = 100,
    support_extension: Union[int, float] = 10,
    precision: int = 1_000,
    alpha: float = 1e-5,
    return_params: bool = True,
):
    """Get parameters via the histogram transform
    
    Parameters
    ----------
    X : jnp.ndarray, (n_samples)
        input to get histogram transformation
    
    support_extension: Union[int, float], default=10
        extend the support by x on both sides
    
    precision: int, default=1_000
        the number of points to use for the interpolation
    
    alpha: float, default=1e-5
        the regularization for the histogram. ensures that
        there are no zeros in the empirical pdf.
    
    Returns
    -------
    X_trans : jnp.ndarray, (n_samples,)
        the data transformed via the empirical function
    log_dX : jnp.ndarray, (n_samples,)
        the log pdf of the data
    Params: namedTuple
        a named tuple with the elements needed for the
        forward and inverse transformation
    
    Examples
    --------
    >>> # single set of parameters
    >>> X_transform, params = get_params(x_samples, 10, 1000)
    
    >>> # example with multiple dimensions
    >>> multi_dims = jax.vmap(get_params, in_axes=(0, None, None))
    >>> X_transform, params = multi_dims(X, 10, 1000)
    """
    # get number of samples
    n_samples = jnp.shape(X)[0]

    # get histogram counts and bin edges
    counts, bin_edges = jnp.histogram(X, bins=nbins)

    # add regularization
    counts = jnp.array(counts) + alpha

    # get bin centers and sizes
    bin_centers = jnp.mean(np.vstack((bin_edges[0:-1], bin_edges[1:])), axis=0)
    bin_size = bin_edges[2] - bin_edges[1]

    # =================================
    # PDF Estimation
    # =================================
    # pdf support
    pdf_support = jnp.hstack(
        (bin_centers[0] - bin_size, bin_centers, bin_centers[-1] + bin_size)
    )
    # empirical PDF
    empirical_pdf = jnp.hstack((0.0, counts / (np.sum(counts) * bin_size), 0.0))

    # =================================
    # CDF Estimation
    # =================================
    c_sum = jnp.cumsum(counts)
    cdf = (1 - 1 / n_samples) * c_sum / n_samples

    incr_bin = bin_size / 2

    # ===============================
    # Extend CDF Support
    # ===============================
    lb, ub = get_domain_extension(X, support_extension)

    # get new bin edges
    new_bin_edges = jnp.hstack((lb, jnp.min(X), bin_centers + incr_bin, ub,))

    extended_cdf = jnp.hstack((0.0, 1.0 / n_samples, cdf, 1.0))

    new_support = jnp.linspace(new_bin_edges[0], new_bin_edges[-1], int(precision))

    uniform_cdf = jax.lax.cummax(
        jnp.interp(new_support, new_bin_edges, extended_cdf), axis=0
    )

    # Normalize CDF estimation
    uniform_cdf /= jnp.max(uniform_cdf)

    # forward transformation
    outputs = jnp.interp(X, new_support, uniform_cdf)

    if return_params is True:

        # initialize parameters
        bijector = UniHistogramTransform(
            support=new_support,
            quantiles=uniform_cdf,
            support_pdf=pdf_support,
            empirical_pdf=empirical_pdf,
        )

        return outputs, bijector
    else:
        return outputs


@dataclass
class UniHistParams:
    support: Array
    quantiles: Array
    support_pdf: Array
    empirical_pdf: Array


def InitUniHistUniformize(
    n_samples: int,
    nbins: Optional[int] = None,
    support_extension: Union[int, float] = 10,
    precision: int = 1_000,
    alpha: float = 1e-5,
):

    # TODO a bin initialization function
    if nbins is None:
        nbins = int(np.sqrt(n_samples))

    def init_fun(inputs):

        outputs, params = get_hist_params(
            X=inputs,
            nbins=nbins,
            support_extension=support_extension,
            precision=precision,
            alpha=alpha,
            return_params=True,
        )

        return outputs, params

    def forward_transform(params, inputs):

        return hist_forward_transform(params, inputs)

    def gradient_transform(params, inputs):

        outputs = forward_transform(params, inputs)

        absdet = hist_gradient_transform(params, inputs)

        logabsdet = jnp.log(absdet)

        return outputs, logabsdet

    def inverse_transform(params, inputs):
        return hist_inverse_transform(params, inputs)

    return init_fun, forward_transform, gradient_transform, inverse_transform


def get_hist_params(
    X: jnp.ndarray,
    nbins: int = 100,
    support_extension: Union[int, float] = 10,
    precision: int = 1_000,
    alpha: float = 1e-5,
    return_params: bool = True,
):
    """Get parameters via the histogram transform
    
    Parameters
    ----------
    X : jnp.ndarray, (n_samples)
        input to get histogram transformation
    
    support_extension: Union[int, float], default=10
        extend the support by x on both sides
    
    precision: int, default=1_000
        the number of points to use for the interpolation
    
    alpha: float, default=1e-5
        the regularization for the histogram. ensures that
        there are no zeros in the empirical pdf.
    
    Returns
    -------
    X_trans : jnp.ndarray, (n_samples,)
        the data transformed via the empirical function
    log_dX : jnp.ndarray, (n_samples,)
        the log pdf of the data
    Params: namedTuple
        a named tuple with the elements needed for the
        forward and inverse transformation
    
    Examples
    --------
    >>> # single set of parameters
    >>> X_transform, params = get_params(x_samples, 10, 1000)
    
    >>> # example with multiple dimensions
    >>> multi_dims = jax.vmap(get_params, in_axes=(0, None, None))
    >>> X_transform, params = multi_dims(X, 10, 1000)
    """
    # get number of samples
    n_samples = jnp.shape(X)[0]

    # get histogram counts and bin edges
    counts, bin_edges = jnp.histogram(X, bins=nbins)

    # add regularization
    counts = jnp.array(counts) + alpha

    # get bin centers and sizes
    bin_centers = jnp.mean(np.vstack((bin_edges[0:-1], bin_edges[1:])), axis=0)
    bin_size = bin_edges[2] - bin_edges[1]

    # =================================
    # PDF Estimation
    # =================================
    # pdf support
    pdf_support = jnp.hstack(
        (bin_centers[0] - bin_size, bin_centers, bin_centers[-1] + bin_size)
    )
    # empirical PDF
    empirical_pdf = jnp.hstack((0.0, counts / (np.sum(counts) * bin_size), 0.0))

    # =================================
    # CDF Estimation
    # =================================
    c_sum = jnp.cumsum(counts)
    cdf = (1 - 1 / n_samples) * c_sum / n_samples

    incr_bin = bin_size / 2

    # ===============================
    # Extend CDF Support
    # ===============================
    lb, ub = get_domain_extension(X, support_extension)

    # get new bin edges
    new_bin_edges = jnp.hstack((lb, jnp.min(X), bin_centers + incr_bin, ub,))

    extended_cdf = jnp.hstack((0.0, 1.0 / n_samples, cdf, 1.0))

    new_support = jnp.linspace(new_bin_edges[0], new_bin_edges[-1], int(precision))

    uniform_cdf = jax.lax.cummax(
        jnp.interp(new_support, new_bin_edges, extended_cdf), axis=0
    )

    # Normalize CDF estimation
    uniform_cdf /= jnp.max(uniform_cdf)

    # forward transformation
    outputs = jnp.interp(X, new_support, uniform_cdf)

    if return_params is True:

        # initialize parameters
        params = UniHistParams(
            support=new_support,
            quantiles=uniform_cdf,
            support_pdf=pdf_support,
            empirical_pdf=empirical_pdf,
        )

        return outputs, params
    else:
        return outputs


def hist_forward_transform(params: UniHistParams, X: JaxArray):
    """Forward univariate uniformize transformation
    
    Parameters
    ----------
    X : jnp.ndarray
        The univariate data to be transformed.
    
    params: UniParams
        the tuple containing the params. 
        See `rbig_jax.transforms.uniformize` for details.
    
    Returns
    -------
    X_trans : jnp.ndarray
        The transformed univariate parameters
    """
    return jnp.interp(X, params.support, params.quantiles)


def hist_inverse_transform(params: UniHistParams, X: JaxArray) -> jnp.ndarray:
    """Inverse univariate uniformize transformation
    
    Parameters
    ----------
    X : jnp.ndarray
        The uniform univariate data to be transformed.
    
    params: UniParams
        the tuple containing the params. 
        See `rbig_jax.transforms.histogram` for details.
    
    Returns
    -------
    X_trans : jnp.ndarray
        The transformed univariate parameters
    """
    return jnp.interp(X, params.quantiles, params.support)


def hist_gradient_transform(params: UniHistParams, X: JaxArray) -> jnp.ndarray:
    """Forward univariate uniformize transformation gradient
    
    Parameters
    ----------
    X : jnp.ndarray
        The univariate data to be transformed.
    
    params: UniParams
        the tuple containing the params. 
        See `rbig_jax.transforms.histogram` for details.
    
    Returns
    -------
    X_trans : jnp.ndarray
        The transformed univariate parameters
    """
    return jnp.interp(X, params.support_pdf, params.empirical_pdf)
