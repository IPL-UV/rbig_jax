from typing import Callable, List, NamedTuple, Tuple, Union

import chex
import jax
from chex import Array, dataclass


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
