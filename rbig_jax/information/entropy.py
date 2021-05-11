from typing import Callable, Optional

import jax
import jax.numpy as np
from chex import Array


def histogram_entropy(data, nbins: int = 10):
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
    S = univariate_entropy(pk)

    # Miller Maddow Correction
    correction = 0.5 * (np.sum(counts > 0) - 1) / counts.sum()

    return S + correction + np.log2(delta)


def init_marginal_histogram_entropy(nbins: int = 10):
    """A wrapper to create a marginal histogram transformation.
    This makes it easier for jit compilation and it also
    parallelizes the computations via `vmap`.

    Parameters
    ----------
    base : int, optional
        [description], by default 2
    nbins : int, optional
        [description], by default 10

    Returns
    -------
    f : callable
        a function to be called
    """
    return jax.vmap(jax.partial(histogram_entropy, nbins=nbins))


def univariate_entropy(pk: np.ndarray) -> float:
    """univariate entropies

    Parameters
    ----------
    pk : np.ndarray
        the normalized probabilities

    Returns
    -------
    H : np.ndarray
        the entropy (nats)
    """
    # calculate entropy
    vec = jax.scipy.special.entr(pk)

    # sum all values
    S = np.sum(vec)

    return S


def get_default_entropy(n_samples: int):
    nbins = int(np.sqrt(n_samples))
    entropy_f = jax.partial(histogram_entropy, nbins=nbins)

    return entropy_f


def rbig_multivariate_entropy(X: Array, nbins: Optional[int] = None, **kwargs):

    n_samples = X.shape[0]

    if nbins is None:
        nbins = int(np.sqrt(n_samples))

    # Calculate entropy in data domain
    H_x = jax.vmap(histogram_entropy, in_axes=(1, None))(X, nbins).sum()

    from rbig_jax.information.total_corr import rbig_total_correlation

    # calculate the total correlation
    tc = rbig_total_correlation(X=X, nbins=nbins, return_all=False, **kwargs)
    return H_x - tc
