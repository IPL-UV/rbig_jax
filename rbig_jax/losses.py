from chex import Array, dataclass
import jax
import jax.numpy as jnp
from rbig_jax.information.total_corr import init_information_reduction_loss
from typing import Callable, NamedTuple, Optional


class IterativeLoss(NamedTuple):
    loss_f: Callable
    condition: Callable
    state: dataclass


class IterativeInfoLoss(NamedTuple):
    loss_f: Callable
    condition: Callable
    state: dataclass
    name: str = "info"


@dataclass(frozen=True)
class InfoLossState:
    max_layers: int
    ilayer: int
    info_loss: Array

    def update_state(self, info_loss: Array):

        # update info loss
        info_loss = jax.ops.index_update(self.info_loss, self.ilayer, info_loss)

        # create new state
        return InfoLossState(
            max_layers=self.max_layers, ilayer=self.ilayer + 1, info_loss=info_loss,
        )


def init_info_loss(
    max_layers: int = 50,
    zero_tolerance: int = 5,
    n_samples: int = 1_000,
    jitted: bool = True,
    info_loss_f: Optional[Callable] = None,
    **kwargs,
):

    window = jnp.ones(zero_tolerance) / zero_tolerance

    # intialize condition
    def info_loss_condition(state):

        # rolling average of absolute value
        x_cumsum_window = jnp.convolve(jnp.abs(state.info_loss), window, "valid")

        # count number of zeros in moving window
        n_zeros = jnp.sum(jnp.where(x_cumsum_window > 0.0, 0, 1)).astype(jnp.int32)

        return jnp.logical_and(jax.lax.ne(n_zeros, 1), state.ilayer < state.max_layers)

    # intialize loss function
    if info_loss_f is None:
        info_loss_f = init_information_reduction_loss(n_samples=n_samples, **kwargs)

    # jit arguments (faster)
    if jitted:
        info_loss_f = jax.jit(info_loss_f)
        info_loss_condition = jax.jit(info_loss_condition)

    # intialize state
    info_loss_state = InfoLossState(
        max_layers=max_layers, ilayer=0, info_loss=jnp.ones(max_layers)
    )

    # create tuple of entries
    return IterativeInfoLoss(
        loss_f=info_loss_f, condition=info_loss_condition, state=info_loss_state,
    )
