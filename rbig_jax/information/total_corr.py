from collections import namedtuple
from typing import Callable

import jax
import jax.numpy as np

from rbig_jax.information.entropy import histogram_entropy
from rbig_jax.information.rbig import TrainState

RBIGEntropy = namedtuple(
    "RBIGEntropy", ["n_layers", "entropy", "mutual_info", "info_loss", "data"]
)


def get_tolerance_dimensions(n_samples: int):
    xxx = np.logspace(2, 8, 7)
    yyy = np.array([0.1571, 0.0468, 0.0145, 0.0046, 0.0014, 0.0001, 0.00001])
    tol_dimensions = np.interp(n_samples, xxx, yyy)
    return tol_dimensions


def information_reduction(X: np.ndarray, Y: np.ndarray, p: float = 0.25) -> float:
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
    hx = jax.vmap(histogram_entropy)(X.T)
    hy = jax.vmap(histogram_entropy)(Y.T)

    # Information content
    delta_info = np.sum(hy) - np.sum(hx)
    tol_info = np.sqrt(np.sum((hy - hx) ** 2))

    # get tolerance
    n_samples, n_dimensions = X.shape

    tol_dimensions = get_tolerance_dimensions(n_samples)
    cond = np.logical_or(
        tol_info < np.sqrt(n_dimensions * p * tol_dimensions ** 2), delta_info < 0
    )
    return np.where(cond, 0.0, delta_info)


def rbig_total_corr(
    X: np.ndarray,
    func: Callable,
    n_layers: int = 100,
    min_layers: int = 10,
    max_layers: int = 10,
    tol_layers: int = 50,
    threshold: float = 0.25,
) -> RBIGEntropy:

    # initialize training state
    train_state = TrainState(n_layers=0, info_loss=np.zeros((1_000,)), X=X,)

    # set condition function
    def condition_fun(state):
        # stopping criterial
        stop_crit = jax.lax.bitwise_and(
            jax.lax.bitwise_not(state.n_layers < min_layers),
            state.n_layers > max_layers,
        )
        stop_crit = jax.lax.bitwise_not(stop_crit)
        return stop_crit

    # find body function
    def body(train_state):
        Xtrans = func(train_state.X)

        # calculate the information loss
        it = information_reduction(train_state.X, Xtrans)

        return TrainState(
            train_state.n_layers + 1,
            jax.ops.index_update(train_state.info_loss, train_state.n_layers + 1, it),
            Xtrans,
        )

    # loop though
    state = jax.lax.while_loop(condition_fun, body, train_state)

    # calculate entropy
    Hx = jax.vmap(histogram_entropy, in_axes=(0, None))(X.T, 2).sum()

    # calculate mutual info
    mutual_info = np.sum(state[1]) * np.log(2)

    return RBIGEntropy(
        n_layers=state.n_layers,
        info_loss=state.info_loss,
        data=state.X,
        mutual_info=mutual_info,
        entropy=Hx.sum() - mutual_info,
    )
