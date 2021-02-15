import collections

import jax.numpy as np

UniParams = collections.namedtuple(
    "UniParams", ["support", "quantiles", "support_pdf", "empirical_pdf"]
)


def forward_uniformization(X: np.ndarray, params: UniParams) -> np.ndarray:
    """Forward univariate uniformize transformation
    
    Parameters
    ----------
    X : np.ndarray
        The univariate data to be transformed.
    
    params: UniParams
        the tuple containing the params. 
        See `rbig_jax.transforms.histogram` for details.
    
    Returns
    -------
    X_trans : np.ndarray
        The transformed univariate parameters
    """
    return np.interp(X, params.support, params.quantiles)


def inverse_uniformization(X: np.ndarray, params: UniParams) -> np.ndarray:
    """Inverse univariate uniformize transformation
    
    Parameters
    ----------
    X : np.ndarray
        The uniform univariate data to be transformed.
    
    params: UniParams
        the tuple containing the params. 
        See `rbig_jax.transforms.histogram` for details.
    
    Returns
    -------
    X_trans : np.ndarray
        The transformed univariate parameters
    """
    return np.interp(X, params.quantiles, params.support)


def forward_uniformization_gradient(X: np.ndarray, params: UniParams) -> np.ndarray:
    """Forward univariate uniformize transformation gradient
    
    Parameters
    ----------
    X : np.ndarray
        The univariate data to be transformed.
    
    params: UniParams
        the tuple containing the params. 
        See `rbig_jax.transforms.histogram` for details.
    
    Returns
    -------
    X_trans : np.ndarray
        The transformed univariate parameters
    """
    return np.interp(X, params.support_pdf, params.empirical_pdf)
