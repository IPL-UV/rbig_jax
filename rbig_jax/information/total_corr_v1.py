from collections import namedtuple
from typing import Callable
import functools

import jax
import jax.numpy as np

from rbig_jax.information.entropy import histogram_entropy
from rbig_jax.information.rbig import TrainState
from rbig_jax.information.rbig import rbig_init
from rbig_jax.transforms.block import (
    forward_gauss_block_transform,
    inverse_gauss_block_transform,
)
from rbig_jax.stopping import info_red_cond
from

InfoLoss = namedtuple("InfoLoss", ["layer", "loss", "total_corr"])

RBIGEntropy = namedtuple(
    "RBIGEntropy", ["n_layers", "entropy", "total_corr", "info_loss", "data"]
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


def rbig_toal_corr(marginal_ent_f: Callable) -> Tuple[,]:
    return None


def rbig_total_corr_old(
    X: np.ndarray,
    method: str = "histogram",
    support_ext: int = 10,
    precision: int = 1_000,
    alpha: float = 1e-5,
    min_layers: int = 10,
    max_layers: int = 1_000,
    tol_layers: int = 50,
    threshold: float = 0.25,
    base: int = 2,
) -> RBIGEntropy:

    # init RBIG params getter
    rbig_params_init = rbig_init(
        method=method, support_ext=support_ext, precision=precision, alpha=alpha,
    )

    # compile function (faster)
    rbig_params_init = jax.jit(rbig_params_init)
    _ = rbig_params_init(X[:10])

    # init loss with stopping criteria
    init_loss = np.pad(np.zeros((max_layers,)), (tol_layers, 0))
    init_loss = jax.ops.index_update(
        init_loss, np.arange(0, tol_layers, dtype=int), 1.0
    )
    train_state = TrainState(n_layers=0, info_loss=init_loss, X=X)

    # jit a few functions
    information_reduction_jitted = jax.jit(
        functools.partial(information_reduction, p=threshold)
    )

    def condition(state):

        # get relevant layers (moving window)
        layers = state.info_loss[state.n_layers : tol_layers + state.n_layers]
        #     print(layers)
        info_sum = np.sum(np.abs(layers))

        # condition - there needs to be some loss of info
        info_crit = info_sum == 0.0
        verdict = jax.lax.bitwise_not(info_crit)
        #     print(f"Info: {info_sum}, verdict: {verdict}")
        return verdict

    # find body function
    def body(train_state):
        Xtrans = rbig_params_init(train_state.X)

        # calculate the information loss
        it = information_reduction_jitted(train_state.X, Xtrans)

        return TrainState(
            n_layers=train_state.n_layers + 1,
            info_loss=jax.ops.index_update(
                train_state.info_loss, tol_layers + train_state.n_layers, it
            ),
            X=Xtrans,
        )

    body = jax.jit(body)

    # loop though
    while condition(train_state):
        train_state = body(train_state)

    # get credible
    info_loss = train_state.info_loss[tol_layers : tol_layers + train_state.n_layers]
    # calculate entropy
    Hx = jax.vmap(histogram_entropy, in_axes=(0, None))(X.T, base).sum()

    # calculate mutual info
    total_corr = np.sum(info_loss) * np.log(base)

    return RBIGEntropy(
        n_layers=train_state.n_layers,
        info_loss=info_loss,
        data=train_state.X,
        total_corr=total_corr,
        entropy=Hx.sum() - total_corr,
    )


class RBIGTC:
    def __init__(
        self,
        rbig_block: Callable,
        tol_layers: int = 10,
        max_layers: int = 1_000,
        p: float = 0.25,
    ):

        self.block_fit = rbig_block
        self.block_forward = forward_gauss_block_transform
        self.block_inverse = inverse_gauss_block_transform
        self.info_loss = jax.partial(information_reduction, p=p)
        self.max_layers = max_layers
        self.tol_layers = tol_layers

    def fit_transform(self, X):

        self.n_features = X.shape[1]

        # initialize parameter storage
        delta_total_corr = []
        i_layer = 0

        # initialize condition state
        state = (0, delta_total_corr, self.tol_layers, self.max_layers)
        while info_red_cond(state):

            # fix info criteria
            loss_f = jax.partial(self.info_loss, X=X)
            X = self.block_fit(X)

            loss = loss_f(Y=X)

            # append Parameters
            delta_total_corr.append(loss)

            # update the state
            state = (i_layer, delta_total_corr, self.tol_layers, self.max_layers)

            i_layer += 1
        self.n_layers = i_layer
        self.delta_total_corr = np.array(delta_total_corr)
        return X


class RBIGTCJit(RBIGTC):
    def __init__(
        self,
        rbig_block: Callable,
        tol_layers: int = 10,
        max_layers: int = 1_000,
        p: float = 0.25,
    ):

        self.block_fit = jax.jit(rbig_block)
        self.block_forward = jax.jit(forward_gauss_block_transform)
        self.block_inverse = jax.jit(inverse_gauss_block_transform)
        self.info_loss = jax.jit(jax.partial(information_reduction, p=p))
        self.max_layers = max_layers
        self.tol_layers = tol_layers
