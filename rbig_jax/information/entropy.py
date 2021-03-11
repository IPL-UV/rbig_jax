from typing import Callable, Optional

import jax
import jax.numpy as np
from chex import Array


def histogram_entropy(data, base=2, nbins: int = 10):
    """Calculates the histogram entropy of 1D data.
    This function uses the histogram and then calculates
    the entropy. Does the miller-maddow correction
    
    Parameters
    ----------
    data : np.ndarray, (n_samples,)
        the input data for the entropy
    
    base : int, default=2
        the log base for the calculation.
    
    Returns
    -------
    S : float
        the entropy"""

    # get histogram counts and bin edges
    counts, bin_edges = np.histogram(data, bins=nbins, density=False)

    # get bin centers and sizes
    bin_centers = np.mean(np.vstack((bin_edges[0:-1], bin_edges[1:])), axis=0)

    # get difference between the bins
    delta = bin_centers[3] - bin_centers[2]

    # normalize counts (density)
    pk = 1.0 * np.array(counts) / np.sum(counts)

    # calculate the entropy
    S = univariate_entropy(pk, base=base)

    # Miller Maddow Correction
    correction = 0.5 * (np.sum(counts > 0) - 1) / counts.sum()

    return S + correction + np.log2(delta)


def marginal_histogram_entropy_f(data, base: int = 2, nbins: int = 10):
    return jax.vmap(jax.partial(histogram_entropy, base=base, nbins=nbins))


def univariate_entropy(pk: np.ndarray, base: int = 2) -> np.ndarray:
    """calculate the entropy
    
    Notes
    -----
    Source of this module is the scipy entropy
    module which can be found - shorturl.at/pyABR
    """
    # calculate entropy
    vec = jax.scipy.special.entr(pk)

    # sum all values
    S = np.sum(vec)

    # change base
    S /= np.log(base)

    return S


def get_default_entropy(n_samples: int):
    nbins = int(np.sqrt(n_samples))
    entropy_f = jax.partial(histogram_entropy, nbins=nbins, base=2)

    return entropy_f


def rbig_multivariate_entropy(
    X: Array, base: int = 2, nbins: Optional[int] = None, **kwargs
):

    n_samples = X.shape[0]

    if nbins is None:
        nbins = int(np.sqrt(n_samples))

    # Calculate entropy in data domain
    H_x = jax.vmap(histogram_entropy, in_axes=(1, None, None))(X, base, nbins).sum()

    from rbig_jax.information.total_corr import rbig_total_correlation

    # calculate the total correlation
    tc = rbig_total_correlation(X=X, nbins=nbins, base=base, return_all=False, **kwargs)
    return H_x - (np.sum(tc) * np.log(base))
