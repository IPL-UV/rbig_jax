from typing import Callable, List, NamedTuple, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from chex import Array, dataclass
from distrax._src.bijectors.bijector import Bijector as NonTrainableBijector
from flax import struct

from rbig_jax.transforms.base import Bijector


@struct.dataclass
class MarginalUniformizeTransform(Bijector):
    def __init__(
        self, support: Array, quantiles: Array, support_pdf: Array, empirical_pdf: Array
    ):
        self.support = support
        self.quantiles = quantiles
        self.support_pdf = support_pdf
        self.empirical_pdf = empirical_pdf

    def forward(self, inputs: Array) -> Array:
        """Marginal transform given a dataset, function and params
        
        Parameters
        ----------
        X : Array
            the input data to do a marginal transform (dimension-wise) of
            shape=(n_samples, n_features)
        
        Returns
        -------
        X_trans : Array
            the output array for the transform
        
        Examples
        --------
        
        >>> init_hist_f = InitUniHistUniformize(10, 10)
        >>> f = lambda x: x ** 2
        >>> X = np.ones((10, 2,))
        >>> X_u, hist_bijector = init_hist_f.init_bijector(X)
        >>> X_l1 = hist_bijector.forward(X)
        """
        # transformation
        outputs = jax.vmap(jnp.interp, in_axes=(1, 0, 0), out_axes=1)(
            inputs, self.support, self.quantiles
        )

        return outputs

    def inverse(self, inputs: Array) -> Array:

        # inverse transformation
        outputs = jax.vmap(jnp.interp, in_axes=(1, 0, 0), out_axes=1)(
            inputs, self.quantiles, self.support
        )

        return outputs

    def forward_log_det_jacobian(self, inputs: Array) -> Array:

        # transformation
        log_abs_det = jax.vmap(jnp.interp, in_axes=(1, 0, 0), out_axes=1)(
            inputs, self.support_pdf, self.empirical_pdf
        )

        log_abs_det = jnp.log(log_abs_det)

        return log_abs_det

    def forward_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:

        # transformation
        outputs = jax.vmap(jnp.interp, in_axes=(1, 0, 0), out_axes=1)(
            inputs, self.support, self.quantiles
        )

        # gradient transformation
        log_abs_det = jax.vmap(jnp.interp, in_axes=(1, 0, 0), out_axes=1)(
            inputs, self.support_pdf, self.empirical_pdf
        )

        log_abs_det = jnp.log(log_abs_det)

        return outputs, log_abs_det

    def inverse_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:

        # inverse transformation
        outputs = jax.vmap(jnp.interp, in_axes=(1, 0, 0), out_axes=1)(
            inputs, self.quantiles, self.support
        )

        # gradient transformation
        log_abs_det = jax.vmap(jnp.interp, in_axes=(1, 0, 0), out_axes=1)(
            inputs, self.support_pdf, self.empirical_pdf
        )

        log_abs_det = jnp.log(log_abs_det)

        return outputs, log_abs_det


def marginal_fit_transform(X: Array, f: Callable) -> Tuple[chex.Array, dataclass]:
    """Marginal transform given a dataset and function
    
    Parameters
    ----------
    X : Array
        the input data to do a marginal transform (dimension-wise) of
        shape=(n_samples, n_features)
    f : Callable[[Array], Tuple[Array, dataclass]]
        the function to be called on the input data
    
    Returns
    -------
    X_trans : Array
        the output array for the transform
    params : dataclass
        the output parameters generated from the function
    
    Examples
    --------
    
    >>> init_hist_f = InitUniHistUniformize(10, 10)
    >>> X = np.ones((10, 2,))
    >>> X_trans, params = marginal_fit_transform(X, init_hist_f)
    """
    X, params = jax.vmap(f, out_axes=(1, 0), in_axes=(1,))(X)
    return X, params


def marginal_transform(X, params: dataclass, f: Callable) -> Array:
    """Marginal transform given a dataset, function and params
    
    Parameters
    ----------
    X : Array
        the input data to do a marginal transform (dimension-wise) of
        shape=(n_samples, n_features)
    params : dataclass
        the params to be passed into the function
    f : Callable[[dataclass, Array], Array]
        the function to be called on the input data
    
    Returns
    -------
    X_trans : Array
        the output array for the transform
    
    Examples
    --------
    
    >>> init_hist_f = InitUniHistUniformize(10, 10)
    >>> f = lambda x: x ** 2
    >>> X = np.ones((10, 2,))
    >>> X_trans, params = marginal_fit_transform(X, init_hist_f)
    >>> X_trans = marginal_transform(X, params, f)
    """
    X = jax.vmap(f, in_axes=(0, 1), out_axes=1)(params, X)
    return X


def marginal_gradient_transform(X, params: dataclass, f: Callable) -> Array:
    """Marginal transform given a dataset, function and params
    
    Parameters
    ----------
    X : Array
        the input data to do a marginal transform (dimension-wise) of
        shape=(n_samples, n_features)
    params : dataclass
        the params to be passed into the function
    f : Callable[[dataclass, Array], Tuple[Array, Array]]
        the function to be called on the input data
    
    Returns
    -------
    X_trans : Array
        the output array for the transform
    
    Examples
    --------
    
    >>> init_hist_f = InitUniHistUniformize(10, 10)
    >>> invf = lambda x: np.sqrt(x)
    >>> X = np.ones((10, 2,))
    >>> X_trans, params = marginal_fit_transform(X, init_hist_f)
    >>> X_trans = marginal_transform(X, params, invf)
    """
    X, X_ldj = jax.vmap(f, in_axes=(0, 1), out_axes=(1, 1))(params, X)
    return X, X_ldj
