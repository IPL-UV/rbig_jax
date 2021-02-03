from typing import Callable, Union, Tuple
import jax
import jax.numpy as np


def marginal_transform(f: Callable):
    return jax.vmap(f)


def get_domain_extension(
    data: np.ndarray, extension: Union[float, int],
) -> Tuple[float, float]:
    """Gets the extension for the support
    
    Parameters
    ----------
    data : np.ndarray
        the input data to get max and minimum

    extension : Union[float, int]
        the extension
    
    Returns
    -------
    lb : float
        the new extended lower bound for the data
    ub : float
        the new extended upper bound for the data
    """

    # case of int, convert to float
    if isinstance(extension, int):
        extension = float(extension / 100)

    # get the domain
    domain = np.abs(np.max(data) - np.min(data))

    # extend the domain
    domain_ext = extension * domain

    # get the extended domain
    lb = np.min(data) - domain_ext
    up = np.max(data) + domain_ext

    return lb, up


def interp_dim(x_new, x, y):
    return jax.vmap(np.interp, in_axes=(0, 0, 0))(x_new, x, y)


def searchsorted(bin_locations, inputs, eps=1e-6):
    # add noise to prevent zeros
    # bin_locations = bin_locations[..., -1] + eps
    bin_locations = bin_locations + eps

    # find bin locations (parallel bisection search)

    # sum dim
    print("Bins:", bin_locations.shape)
    print("Inputs:", inputs[..., None].shape)
    input_bins = np.sum(inputs[..., None] >= bin_locations, axis=-1)

    return input_bins


