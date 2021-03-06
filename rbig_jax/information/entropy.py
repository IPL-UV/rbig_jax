from typing import Callable

import jax
import jax.numpy as np
from chex import Array

# from rbig_jax.information.total_corr import rbig_total_correlation


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


# def rbig_entropy(
#     X_samples: Array,
#     marginal_uni: Callable,
#     marginal_entropy: Callable,
#     n_iterations: int = 100,
#     base: int = 2,
# ):

#     # create marginal entropy equation
#     marginal_entropy_vectorized = jax.vmap(marginal_entropy)

#     # Calculate entropy in data domain
#     H_x = marginal_entropy_vectorized(X_samples).sum()

#     # calculate the total correlation
#     _, tc = rbig_total_correlation(
#         X_samples,
#         marginal_uni=marginal_uni,
#         marginal_entropy=marginal_entropy,
#         n_iterations=n_iterations,
#     )
#     return H_x - (np.sum(tc) * np.log(base))
