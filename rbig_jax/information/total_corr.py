import functools
from collections import namedtuple
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as np
from chex import Array, dataclass

from rbig_jax.custom_types import InputData
from rbig_jax.information.entropy import histogram_entropy
from rbig_jax.transforms.block import InitRBIGBlock
from rbig_jax.transforms.histogram import InitUniHistUniformize
from rbig_jax.transforms.inversecdf import invgausscdf_forward_transform
from rbig_jax.transforms.linear import svd_transform
from rbig_jax.transforms.rotation import InitPCARotation


@dataclass
class InfoLossState:
    max_layers: int
    ilayer: int
    info_loss: Array


def get_tolerance_dimensions(n_samples: int) -> int:
    xxx = np.logspace(2, 8, 7)
    yyy = np.array([0.1571, 0.0468, 0.0145, 0.0046, 0.0014, 0.0001, 0.00001])
    tol_dimensions = np.interp(n_samples, xxx, yyy)
    return tol_dimensions


def init_information_reduction_loss(
    n_samples: int, nbins: Optional[int] = None, p: int = 0.25, **kwargs
):
    if nbins is None:
        nbins = int(np.sqrt(n_samples))  # important parameter
    entropy_est = jax.partial(histogram_entropy, nbins=nbins, **kwargs)

    tol_dims = get_tolerance_dimensions(n_samples)

    def loss_function(X_before, X_after):

        return information_reduction(
            X_before, X_after, uni_entropy=entropy_est, tol_dims=tol_dims, p=p
        )

    return loss_function


def information_reduction(
    X: np.ndarray, Y: np.ndarray, uni_entropy: Callable, tol_dims: int, p: float = 0.25,
) -> float:
    """calculates the information reduction between layers
    This function computes the multi-information (total correlation)
    reduction after a linear transformation.
    
    .. math::
        Y = XW \\
        II = I(X) - I(Y)
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Data before the transformation, where n_samples is the number
        of samples and n_features is the number of features.
    
    Y : np.ndarray, shape (n_samples, n_features)
        Data after the transformation, where n_samples is the number
        of samples and n_features is the number of features
        
    p : float, default=0.25
        Tolerance on the minimum multi-information difference
        
    Returns
    -------
    II : float
        The change in multi-information
        
    Information
    -----------
    Author: Valero Laparra
            Juan Emmanuel Johnson
    """
    # calculate the marginal entropy
    hx = jax.vmap(uni_entropy)(X.T)
    hy = jax.vmap(uni_entropy)(Y.T)

    # Information content
    delta_info = np.sum(hy) - np.sum(hx)
    tol_info = np.sqrt(np.sum((hy - hx) ** 2))

    # get tolerance
    n_dimensions = X.shape[1]

    # conditional
    cond = np.logical_or(
        tol_info < np.sqrt(n_dimensions * p * tol_dims ** 2), delta_info < 0
    )
    return np.array(np.where(cond, 0.0, delta_info))


def total_correlation(
    X: Array,
    uni_uniformize: Optional[Callable] = None,
    rot_transform: Optional[Callable] = None,
    eps: float = 1e-5,
    zero_tolerance: int = 30,
    max_layers: int = 1_000,
    p: float = 0.25,
    base: int = 2,
    return_all: bool = False,
):

    # create Gaussinization block
    fit_transform_f, *_ = InitRBIGBlock(uni_uniformize, rot_transform, eps)
    block_fit_transform = jax.jit(fit_transform_f)
    # initialize information loss function
    info_loss_f = jax.jit(
        init_information_reduction_loss(n_samples=X.shape[0], base=base, p=p)
    )

    window = np.ones(zero_tolerance) / zero_tolerance

    def condition(state):

        # rolling average
        x_cumsum_window = np.convolve(np.abs(state.info_loss), window, "valid")
        n_zeros = int(np.sum(np.where(x_cumsum_window > 0.0, 0, 1)))
        return jax.lax.ne(n_zeros, 1) or state.ilayer > state.max_layers

    # initialize loss states
    state = InfoLossState(
        max_layers=max_layers, ilayer=0, info_loss=np.ones(max_layers)
    )

    X_g = X
    while condition(state):

        layer_loss = jax.partial(info_loss_f, X_before=X_g)

        # compute
        X_g, layer_params = block_fit_transform(X_g)

        # get information reduction
        layer_loss = layer_loss(X_after=X_g)

        # update layer loss
        info_losses = jax.ops.index_update(state.info_loss, state.ilayer, layer_loss)
        state = InfoLossState(
            max_layers=max_layers, ilayer=state.ilayer + 1, info_loss=info_losses,
        )

    info_loss = info_losses[: state.ilayer]
    if return_all:
        return X_g, info_loss
    else:
        return np.sum(info_loss) * np.log(base)


def total_correlation_max_layers(
    X: Array,
    uni_uniformize: Optional[Callable] = None,
    rot_transform: Optional[Callable] = None,
    eps: float = 1e-5,
    max_layers: int = 1_000,
    p: float = 0.25,
    return_all: bool = False,
):

    # create Gaussinization block
    fit_transform_f, *_ = InitRBIGBlock(uni_uniformize, rot_transform, eps)
    block_fit_transform = jax.jit(fit_transform_f)
    # initialize information loss function
    info_loss_f = jax.jit(
        init_information_reduction_loss(n_samples=X.shape[0], base=2, p=p)
    )

    def body(carry, inputs):
        layer_loss = jax.partial(info_loss_f, X_before=carry)

        # do transformation
        carry, _ = block_fit_transform(carry)

        layer_loss = layer_loss(X_after=carry)
        carry = np.array(carry, dtype=np.float32)
        return carry, layer_loss

    X, info_loss = jax.lax.scan(f=body, init=X, xs=None, length=max_layers)

    if return_all:
        return X, info_loss
    else:
        return np.sum(info_loss)


def rbig_total_correlation(
    X: Array,
    nbins: Optional[int] = None,
    precision: int = 100,
    support_extension: int = 10,
    alpha: int = 1e-5,
    **kwargs,
):

    n_samples = X.shape[0]
    if nbins is None:
        nbins = int(np.sqrt(n_samples))
    uni_uniformize = InitUniHistUniformize(
        n_samples=n_samples,
        nbins=nbins,
        support_extension=support_extension,
        precision=precision,
        alpha=alpha,
    )

    rot_transform = InitPCARotation()

    return total_correlation(
        X=X, uni_uniformize=uni_uniformize, rot_transform=rot_transform, **kwargs
    )
