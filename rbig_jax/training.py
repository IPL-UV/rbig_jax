import itertools
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from chex import dataclass
from distrax._src.distributions import distribution as dist_base
from jax import scipy as jscipy
from distrax._src.utils.math import sum_last

DistributionLike = dist_base.DistributionLike


def init_log_prob(base_dist: DistributionLike) -> Callable:
    def log_prob(bijector, inputs):

        # forward transformation
        outputs, log_det = bijector.forward_and_log_det(inputs)

        # probability in the latent space
        latent_prob = base_dist.log_prob(outputs)

        # log probability
        log_prob = sum_last(latent_prob, ndims=latent_prob.ndim - 1) + sum_last(
            log_det, ndims=log_det.ndim - 1
        )

        # # log probability
        # log_prob = sum_last(latent_prob, ndims=latent_prob.ndim) + sum_last(
        #     log_det, ndims=latent_prob.ndim
        # )
        return log_prob

    return log_prob


def init_train_op(
    params: dataclass,
    loss_f: Callable,
    optimizer,
    lr: float = 1e-2,
    jitted: bool = True,
):
    # unpack optimizer params
    opt_init, opt_update, get_params = optimizer(step_size=lr)

    # initialize parameters
    opt_state = opt_init(params)

    # create training loops
    def train_op(i, opt_state, inputs):
        # get the parameters from the state
        params = get_params(opt_state)

        # calculate the loss AND the gradients
        loss, gradients = jax.value_and_grad(loss_f)(params, inputs)

        # return loss AND new opt_state
        return loss, opt_update(i, gradients, opt_state)

    if jitted:
        train_op = jax.jit(train_op)
    return train_op, (opt_init, opt_state, get_params)


def train_model(
    train_op: Callable, opt_params: Callable, dl, epochs: int = 100, **kwargs
):
    losses = list()
    itercount = itertools.count()

    _, opt_state, get_params = opt_params

    pbar = tqdm.trange(epochs)

    with pbar:
        for i in pbar:

            # Train
            avg_loss = []

            for ix in dl:

                # cast to jax array
                ix = jnp.array(ix, dtype=jnp.float32)

                # compute loss
                loss, opt_state = train_op(next(itercount), opt_state, ix,)

                # append batch
                avg_loss.append(float(loss))

            # average loss
            batch_loss = jnp.mean(jnp.stack(avg_loss))

            # Log losses
            losses.append(np.array(batch_loss))
            pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

    final_params = get_params(opt_state)
    return final_params, losses

