from collections import namedtuple
from typing import Callable

import jax
import jax.numpy as np

from rbig_jax.information.entropy import histogram_entropy
from rbig_jax.information.rbig import TrainState
from rbig_jax.information.reduction import information_reduction

RBIGMI = namedtuple("RBIGEntropy", ["MIx", "MIy", "mutual_info"])

information_reduction = jax.jit(information_reduction)


def rbig_mutual_info(
    X: np.ndarray,
    Y: np.ndarray,
    func: Callable,
    n_layers: int = 100,
    min_layers: int = 10,
    max_layers: int = 100,
    tol_layers: int = 50,
    threshold: float = 0.25,
) -> RBIGMI:

    # set condition function
    def condition_fun(state):
        # stopping criterial
        stop_crit = jax.lax.bitwise_and(
            jax.lax.bitwise_not(state.n_layers < min_layers),
            state.n_layers > max_layers,
        )
        stop_crit = jax.lax.bitwise_not(stop_crit)
        return stop_crit

    # find body functionrbig
    def body(train_state):
        Xtrans = func(train_state.X)

        # calculate the information loss
        it = information_reduction(train_state.X, Xtrans)

        return TrainState(
            train_state.n_layers + 1,
            jax.ops.index_update(train_state.info_loss, train_state.n_layers + 1, it),
            Xtrans,
        )

    condition_fun = jax.jit(condition_fun)
    body = jax.jit(body)

    # initialize train state
    train_state_X = TrainState(
        n_layers=0, info_loss=np.zeros((max_layers,)), X=np.array(X),
    )

    # find total correlation
    state_X = jax.lax.while_loop(condition_fun, body, train_state_X)

    # initialize train state
    train_state_Y = TrainState(
        n_layers=0, info_loss=np.zeros((max_layers,)), X=np.array(Y),
    )

    # find total correlation
    state_Y = jax.lax.while_loop(condition_fun, body, train_state_Y)

    # initialize train state
    train_state_XY = TrainState(
        n_layers=0,
        info_loss=np.zeros((max_layers,)),
        X=np.hstack([state_X.X, state_Y.X]),
    )

    # find total correlation
    state_XY = jax.lax.while_loop(condition_fun, body, train_state_XY)

    return RBIGMI(
        MIx=np.sum(state_X.info_loss) * np.log(2),
        MIy=np.sum(state_Y.info_loss) * np.log(2),
        mutual_info=np.sum(state_XY.info_loss) * np.log(2),
    )
