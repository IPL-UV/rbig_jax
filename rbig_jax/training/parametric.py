import itertools
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from chex import dataclass
from distrax._src.distributions import distribution as dist_base
from distrax._src.utils.math import sum_last
from jax import scipy as jscipy
from jax.experimental import optimizers

DistributionLike = dist_base.DistributionLike
JAXOptimizer = jax.experimental.optimizers.Optimizer


class TrainingParams(NamedTuple):
    train_op: Callable
    opt_init: Callable
    opt_state: Callable
    get_params: Callable


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


def init_gf_train_op(
    gf_model: dataclass, optimizer, lr: float = 1e-2, jitted: bool = True,
):
    # unpack optimizer params
    opt_init, opt_update, get_params = optimizer(step_size=lr)

    # initialize parameters
    opt_state = opt_init(gf_model)

    def loss_f(gf_model, inputs):
        return gf_model.score(inputs)

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

    return TrainingParams(
        train_op=train_op, opt_init=opt_init, opt_state=opt_state, get_params=get_params
    )


def train_model(
    gf_model: dataclass,
    train_dl,
    valid_dl=None,
    epochs: int = 100,
    optimizer: Optional[JAXOptimizer] = None,
    lr: float = 0.01,
    jitted: bool = True,
    **kwargs,
):

    if optimizer is None:

        optimizer = optimizers.adam(step_size=0.01)

    # unpack optimizer params
    opt_init, opt_update, get_params = optimizer

    # initialize parameters
    opt_state = opt_init(gf_model)

    # define loss function
    def loss_f(gf_model, inputs):
        return gf_model.score(inputs)

    # ================================
    # Boilerplate Code for Training
    # ================================

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

    # ================================
    # TRAINING
    # ================================
    train_losses = list()
    valid_losses = list()
    itercount = itertools.count()
    train_batch_loss = 0.0
    valid_batch_loss = 0.0

    pbar = tqdm.trange(epochs)

    with pbar:
        for _ in pbar:

            # Train
            avg_loss = []

            for ix in train_dl:

                # cast to jax array
                ix = jnp.array(ix, dtype=jnp.float32)

                # compute loss
                loss, opt_state = train_op(next(itercount), opt_state, ix,)

                # append batch
                avg_loss.append(float(loss))

            # average loss
            train_batch_loss = jnp.mean(jnp.stack(avg_loss))

            # Log losses
            train_losses.append(np.array(train_batch_loss))
            pbar.set_postfix(
                {
                    "Train Loss": f"{train_batch_loss:.4f}",
                    "Valid Loss": f"{valid_batch_loss:.4f}",
                }
            )

            if valid_dl is not None:

                final_params = get_params(opt_state)

                # Train
                avg_loss = []

                for ix in valid_dl:

                    # cast to jax array
                    ix = jnp.array(ix, dtype=jnp.float32)

                    # compute loss
                    loss = final_params.score(ix)

                    # append batch
                    avg_loss.append(float(loss))

                # average loss
                valid_batch_loss = jnp.mean(jnp.stack(avg_loss))

                valid_losses.append(np.array(valid_batch_loss))

                pbar.set_postfix(
                    {
                        "Train Loss": f"{train_batch_loss:.4f}",
                        "Valid Loss": f"{valid_batch_loss:.4f}",
                    }
                )

            else:
                continue

    final_params = get_params(opt_state)

    train_losses = jnp.stack(train_losses)
    if valid_dl is not None:
        valid_losses = jnp.stack(valid_losses)
    else:
        valid_losses = None
    losses = {"train": train_losses, "valid": valid_losses}
    return final_params, losses


def add_gf_train_args(parser):
    # ====================
    # Model Args
    # ====================
    parser.add_argument(
        "--epochs", type=int, default=100, help="Standardize Input Training Data",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Standardize Input Training Data"
    )
    parser.add_argument(
        "--eval_freq", type=int, default=50, help="Standardize Input Training Data"
    )

    parser.add_argument(
        "--optimizer", type=str, default="adam", help="Standardize Input Training Data",
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=1.0,
        help="Standardize Input Training Data",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Standardize Input Training Data",
    )
    return parser
