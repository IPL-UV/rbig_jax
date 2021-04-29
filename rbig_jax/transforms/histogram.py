import collections
import math
from typing import Callable, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from chex import Array, dataclass
from chex._src.pytypes import PRNGKey
from distrax._src.bijectors.bijector import Bijector as distaxBijector

from rbig_jax.transforms.base import InitLayersFunctions, NonTrainableBijector
from rbig_jax.transforms.marginal import MarginalUniformizeTransform
from rbig_jax.utils import get_domain_extension, marginal_transform


class UniHistParams(NamedTuple):

    support: Array
    quantiles: Array
    support_pdf: Array
    empirical_pdf: Array


# class InitUniHistTransform(NonTrainableBijector):
#     def __init__(
#         self,
#         shape: int,
#         nbins: Optional[int] = 10,
#         support_extension: Union[int, float] = 10,
#         precision: int = 1_000,
#         alpha: float = 1e-5,
#         jitted: bool = True,
#     ):
#         self.shape = shape
#         self.support_extension = support_extension
#         self.precision = precision
#         self.alpha = alpha
#         self.nbins = nbins

#         f_params = jax.partial(
#             init_hist_params,
#             nbins=nbins,
#             support_extension=support_extension,
#             precision=precision,
#             alpha=alpha,
#             return_params=True,
#         )
#         f = jax.partial(
#             init_hist_params,
#             nbins=nbins,
#             support_extension=support_extension,
#             precision=precision,
#             alpha=alpha,
#             return_params=False,
#         )
#         if jitted:
#             f = jax.jit(f)
#             f_params = jax.jit(f_params)

#         self._f = f
#         self._f_params = f_params

#     def forward(self, inputs):

#         outputs = jax.vmap(self._f, out_axes=1, in_axes=(1,))(inputs)

#         return outputs

#     def forward_and_log_det(self, inputs):
#         outputs, params = jax.vmap(self._f_params, out_axes=(1, 0), in_axes=(1,))(
#             inputs
#         )

#         # transformation
#         log_abs_det = jax.vmap(jnp.interp, in_axes=(1, 0, 0), out_axes=1)(
#             inputs, params.support_pdf, params.empirical_pdf
#         )

#         log_abs_det = jnp.log(log_abs_det)

#         return outputs, log_abs_det

#     def forward_and_params(self, inputs):

#         # get parameters
#         outputs, params = jax.vmap(self._f_params, out_axes=(1, 0), in_axes=(1,))(
#             inputs
#         )

#         return outputs, params

#     def forward_and_bijector(self, inputs):

#         # get parameters
#         outputs, params = jax.vmap(self._f_params, out_axes=(1, 0), in_axes=(1,))(
#             inputs
#         )

#         # initialize bijector
#         bijector = MarginalUniformizeTransform(
#             support=params.support,
#             quantiles=params.quantiles,
#             support_pdf=params.support_pdf,
#             empirical_pdf=params.empirical_pdf,
#         )
#         return outputs, bijector

#     def forward_log_det_bijector(self, inputs):

#         # get parameters
#         outputs, params = jax.vmap(self._f_params, out_axes=(1, 0), in_axes=(1,))(
#             inputs
#         )

#         # transformation
#         log_abs_det = jax.vmap(jnp.interp, in_axes=(1, 0, 0), out_axes=1)(
#             inputs, params.support_pdf, params.empirical_pdf
#         )

#         log_abs_det = jnp.log(log_abs_det)

#         # initialize bijector
#         bijector = MarginalUniformizeTransform(
#             support=params.support,
#             quantiles=params.quantiles,
#             support_pdf=params.support_pdf,
#             empirical_pdf=params.empirical_pdf,
#         )
#         return outputs, log_abs_det, bijector

#     def bijector(self, inputs):
#         # get parameters
#         _, params = jax.vmap(self._f_params, out_axes=(1, 0), in_axes=(1,))(inputs)

#         # initialize bijector
#         bijector = MarginalUniformizeTransform(
#             support=params.support,
#             quantiles=params.quantiles,
#             support_pdf=params.support_pdf,
#             empirical_pdf=params.empirical_pdf,
#         )
#         return bijector

#     # @jax.jit(static_argnums=(0,))
#     def _transform_and_params(self, inputs):
#         outputs, params = init_hist_params(
#             X=inputs,
#             nbins=self.nbins,
#             support_extension=self.support_extension,
#             precision=self.precision,
#             alpha=self.alpha,
#             return_params=True,
#         )

#         return outputs, params

#     # @jax.jit(static_argnums=(0,))
#     def _transform(self, inputs):
#         outputs = init_hist_params(
#             X=inputs,
#             nbins=self.nbins,
#             support_extension=self.support_extension,
#             precision=self.precision,
#             alpha=self.alpha,
#             return_params=False,
#         )

#         return outputs

# # TODO a bin initialization function
# if nbins is None:
#     nbins = int(jnp.sqrt(n_samples))

# f = jax.partial(
#     init_hist_params,
#     nbins=nbins,
#     support_extension=support_extension,
#     precision=precision,
#     alpha=alpha,
#     return_params=True,
# )

# f_slim = jax.partial(
#     init_hist_params,
#     nbins=nbins,
#     support_extension=support_extension,
#     precision=precision,
#     alpha=alpha,
#     return_params=False,
# )
# if jitted:
#     f = jax.jit(f)
#     f_slim = jax.jit(f_slim)

# def params_and_transform(inputs, **kwargs):

#     outputs, params = jax.vmap(f, out_axes=(1, 0), in_axes=(1,))(inputs)
#     return outputs, params

# def init_params(inputs, **kwargs):

#     _, params = jax.vmap(f, out_axes=(1, 0), in_axes=(1,))(inputs)
#     return params

# def transform(inputs, **kwargs):

#     outputs = jax.vmap(f_slim, out_axes=1, in_axes=(1,))(inputs)
#     return outputs

# def bijector_and_transform(inputs, **kwargs):
#     outputs, params = jax.vmap(f, out_axes=(1, 0), in_axes=(1,))(inputs)

#     bijector = MarginalUniformizeTransform(
#         support=params.support,
#         quantiles=params.quantiles,
#         support_pdf=params.support_pdf,
#         empirical_pdf=params.empirical_pdf,
#     )
#     return outputs, bijector

# def bijector(X, **kwargs):
#     _, params = jax.vmap(f, out_axes=(1, 0), in_axes=(1,))(X)

#     bijector = MarginalUniformizeTransform(
#         support=params.support,
#         quantiles=params.quantiles,
#         support_pdf=params.support_pdf,
#         empirical_pdf=params.empirical_pdf,
#     )
#     return bijector

# return InitLayersFunctions(
#     bijector=bijector,
#     bijector_and_transform=bijector_and_transform,
#     transform=transform,
#     params=init_params,
#     params_and_transform=params_and_transform,
# )


def InitUniHistTransform(
    n_samples: int,
    nbins: Optional[int] = None,
    support_extension: Union[int, float] = 10,
    precision: int = 1_000,
    alpha: float = 1e-5,
    jitted=False,
):

    # TODO a bin initialization function
    if nbins is None:
        nbins = int(jnp.sqrt(n_samples))

    f = jax.partial(
        init_hist_params,
        nbins=nbins,
        support_extension=support_extension,
        precision=precision,
        alpha=alpha,
        return_params=True,
    )

    f_slim = jax.partial(
        init_hist_params,
        nbins=nbins,
        support_extension=support_extension,
        precision=precision,
        alpha=alpha,
        return_params=False,
    )
    if jitted:
        f = jax.jit(f)
        f_slim = jax.jit(f_slim)

    def params_and_transform(inputs, **kwargs):

        outputs, params = jax.vmap(f, out_axes=(1, 0), in_axes=(1,))(inputs)
        return outputs, params

    def init_params(inputs, **kwargs):

        _, params = jax.vmap(f, out_axes=(1, 0), in_axes=(1,))(inputs)
        return params

    def transform(inputs, **kwargs):

        outputs = jax.vmap(f_slim, out_axes=1, in_axes=(1,))(inputs)
        return outputs

    def bijector_and_transform(inputs, **kwargs):
        outputs, params = jax.vmap(f, out_axes=(1, 0), in_axes=(1,))(inputs)

        bijector = MarginalUniformizeTransform(
            support=params.support,
            quantiles=params.quantiles,
            support_pdf=params.support_pdf,
            empirical_pdf=params.empirical_pdf,
        )
        return outputs, bijector

    def bijector(X, **kwargs):
        _, params = jax.vmap(f, out_axes=(1, 0), in_axes=(1,))(X)

        bijector = MarginalUniformizeTransform(
            support=params.support,
            quantiles=params.quantiles,
            support_pdf=params.support_pdf,
            empirical_pdf=params.empirical_pdf,
        )
        return bijector

    return InitLayersFunctions(
        bijector=bijector,
        bijector_and_transform=bijector_and_transform,
        transform=transform,
        params=init_params,
        params_and_transform=params_and_transform,
    )


def init_hist_params(
    X: Array,
    nbins: int = 100,
    support_extension: Union[int, float] = 10,
    precision: int = 1_000,
    alpha: float = 1e-5,
    return_params: bool = True,
):
    """Get parameters via the histogram transform
    
    Parameters
    ----------
    X : Array, (n_samples)
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
    X_trans : Array, (n_samples,)
        the data transformed via the empirical function
    log_dX : Array, (n_samples,)
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

    # nbins_ = nbins(X)

    # get histogram counts and bin edges
    counts, bin_edges = jnp.histogram(X, bins=nbins)

    # add regularization
    counts = jnp.array(counts) + alpha

    # get bin centers and sizes
    bin_centers = jnp.mean(jnp.vstack((bin_edges[0:-1], bin_edges[1:])), axis=0)
    bin_size = bin_edges[2] - bin_edges[1]

    # =================================
    # PDF Estimation
    # =================================
    # pdf support
    pdf_support = jnp.hstack(
        (bin_centers[0] - bin_size, bin_centers, bin_centers[-1] + bin_size)
    )
    # empirical PDF
    empirical_pdf = jnp.hstack((0.0, counts / (jnp.sum(counts) * bin_size), 0.0))

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

        params = UniHistParams(
            support=new_support,
            quantiles=uniform_cdf,
            support_pdf=pdf_support,
            empirical_pdf=empirical_pdf,
        )

        return outputs, params
    else:
        return outputs


def hist_forward_transform(params: UniHistParams, X: Array):
    """Forward univariate uniformize transformation
    
    Parameters
    ----------
    X : Array
        The univariate data to be transformed.
    
    params: UniParams
        the tuple containing the params. 
        See `rbig_jax.transforms.uniformize` for details.
    
    Returns
    -------
    X_trans : Array
        The transformed univariate parameters
    """
    return jnp.interp(X, params.support, params.quantiles)


def hist_inverse_transform(params: UniHistParams, X: Array) -> Array:
    """Inverse univariate uniformize transformation
    
    Parameters
    ----------
    X : Array
        The uniform univariate data to be transformed.
    
    params: UniParams
        the tuple containing the params. 
        See `rbig_jax.transforms.histogram` for details.
    
    Returns
    -------
    X_trans : Array
        The transformed univariate parameters
    """
    return jnp.interp(X, params.quantiles, params.support)


def hist_gradient_transform(params: UniHistParams, X: Array) -> Array:
    """Forward univariate uniformize transformation gradient
    
    Parameters
    ----------
    X : Array
        The univariate data to be transformed.
    
    params: UniParams
        the tuple containing the params. 
        See `rbig_jax.transforms.histogram` for details.
    
    Returns
    -------
    X_trans : Array
        The transformed univariate parameters
    """
    return jnp.interp(X, params.support_pdf, params.empirical_pdf)


def init_bin_estimator(method="sturges"):
    if method == "sqrt":
        return hist_bin_sqrt
    elif method == "scott":
        raise NotImplementedError(f"Error with data dependent method")

    # return hist_bin_scott
    elif method == "sturges":
        return hist_bin_sturges
    elif method == "rice":
        return hist_bin_rice
    elif method == "fd":
        raise NotImplementedError(f"Error with data dependent method")
        # return hist_bin_fd
    elif method == "auto":
        raise NotImplementedError(f"Error with data dependent method")
        # return hist_bin_auto
    else:
        raise ValueError(f"Unrecognized bin estimation method: {method}")


def _ptp(x):
    return jnp.ptp(x).astype(jnp.uint32)




def hist_bin_sqrt(x):
    return math.ceil(math.sqrt(x.size))


def hist_bin_scott(x):
    return (24.0 * math.pi ** 0.5 / x.size) ** (1.0 / 3.0) * jnp.std(x)


def hist_bin_sturges(x):
    # return _ptp(x) / (jnp.log2(x.size) + 1.0)
    return math.ceil(math.log2(x.size) + 1.0)


def hist_bin_rice(x):
    # return _ptp(x) / (2.0 * x.size ** (1.0 / 3))
    return math.ceil(2.0 * x.size ** (1.0 / 3))


def hist_bin_fd(x):
    iqr = jnp.subtract(*jnp.percentile(x, [75, 25]))
    return 2.0 * iqr * x.size ** (-1.0 / 3.0)


def hist_bin_auto(x):
    fd_bw = hist_bin_fd(x)
    sturges_bw = hist_bin_sturges(x)
    min_bin = jnp.minimum(fd_bw, sturges_bw)
    # if fd 0, use sturges
    return jnp.where(min_bin > 0, min_bin, sturges_bw)
