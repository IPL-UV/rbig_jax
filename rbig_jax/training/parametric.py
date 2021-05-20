import itertools
from typing import Callable, NamedTuple, Optional, Any, Tuple, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import optax
from chex import dataclass, Array
from distrax._src.distributions import distribution as dist_base
from distrax._src.utils.math import sum_last
from jax import scipy as jscipy

DistributionLike = dist_base.DistributionLike

OptState = Any
Batch = Mapping[str, np.ndarray]


class TrainState(NamedTuple):
    model: dataclass
    opt_state: OptState


class StepOutput(NamedTuple):
    loss: Array
    model: dataclass


class GaussFlowTrainer:
    def __init__(
        self, model, optimizer, n_epochs: int = 5_000, prepare_data_fn: Callable = None
    ):
        self.model = model
        self.n_epochs = n_epochs
        self.prepare_data_fn = prepare_data_fn

        opt_state = optimizer.init(model)
        self.optimizer = optimizer
        self.steps = 0.0
        self.counter = itertools.count()

        # init metrics
        self.train_epoch = []
        self.train_loss = []
        self.valid_epoch = []
        self.valid_loss = []

        self.train_state = TrainState(model=model, opt_state=opt_state)

    @jax.partial(jax.jit, static_argnums=(0,))
    def params_update(
        self, params, opt_state, batch: Batch, rng=None
    ) -> Tuple[dataclass, OptState]:
        """Single SGD update step."""
        # calculate the loss AND the gradients
        loss, grads = jax.value_and_grad(self.loss_fn)(params, batch, rng)

        # update the gradients
        updates, new_opt_state = self.optimizer.update(grads, opt_state)

        # update the parameters
        new_model = optax.apply_updates(params, updates)

        # return loss AND new opt_state

        return new_model, new_opt_state, loss

    def train_step(self, data, **kwargs):

        # get params from train state
        model = self.train_state.model
        opt_state = self.train_state.opt_state

        # do a gradient step
        model, opt_state, train_loss = self.params_update(
            model, opt_state, data, **kwargs
        )

        self.train_state = TrainState(model=model, opt_state=opt_state)

        # append to loss
        self.train_loss.append(train_loss)
        self.step = next(self.counter)
        self.train_epoch.append(self.step)

        return StepOutput(model=model, loss=train_loss)

    def validation_step(self, data):

        # get params from train state
        model = self.train_state.model

        # do a gradient step
        valid_loss = self.eval_fn(model, data)

        self.valid_loss.append(valid_loss)
        self.valid_epoch.append(self.step)

        return StepOutput(model=model, loss=valid_loss)

    def train_loop(self, model, train_ds, val_ds=None):

        train_state = []

        # initialize optimizer state
        opt_state = self.optimizer.init(model)

        with tqdm.trange(self.n_epochs) as pbar:

            for step in pbar:

                model, opt_state, train_loss = self.update(model, opt_state, train_ds)

                pbar.set_description(
                    f"Train Loss: {train_loss:.4f} | Valid Loss: {eval_loss:.4f}"
                )

                # save metrics
                self.train_step.append(step)
                self.train_loss.append(train_loss)

                if step % self.eval_frequency == 0 and val_ds is not None:

                    eval_loss = self.eval_fn(model, next(val_ds))

                    pbar.set_description(
                        f"Train Loss: {train_loss:.4f} | Valid Loss: {eval_loss:.4f}"
                    )

                    # save metrics
                    self.valid_step.append(step)
                    self.valid_loss.append(eval_loss)

        return model

    def loss_fn(self, model, batch, rng=None):

        if self.prepare_data_fn is not None:
            batch = self.prepare_data_fn(batch, rng)

        # negative log likelihood loss
        nll_loss = model.score(batch)

        return nll_loss

    @jax.partial(jax.jit, static_argnums=(0,))
    def eval_fn(self, model, batch):
        if self.prepare_data_fn is not None:
            batch = self.prepare_data_fn(batch)
        # negative log likelihood loss
        nll_loss = model.score(batch)

        return nll_loss


class ConditionalGaussFlowTrainer(GaussFlowTrainer):
    def __init__(
        self, cond_model, optimizer, prepare_data_fn: Callable, n_epochs: int = 5_000,
    ):
        super().__init__(
            model=cond_model,
            optimizer=optimizer,
            n_epochs=n_epochs,
            prepare_data_fn=prepare_data_fn,
        )

    def loss_fn(self, model, batch, rng=None):

        inputs, outputs = self.prepare_data_fn(batch, rng)

        # negative log likelihood loss
        nll_loss = model.score(inputs=inputs, outputs=outputs)

        return nll_loss

    @jax.partial(jax.jit, static_argnums=(0,))
    def eval_fn(self, model, batch):

        inputs, outputs = self.prepare_data_fn(batch)

        # negative log likelihood loss
        nll_loss = model.score(inputs=inputs, outputs=outputs)

        return nll_loss


def init_optimizer(
    name: str = "adam",
    n_epochs: int = 5_000,
    lr: float = 1e-3,
    cosine_decay_steps: Optional[int] = None,
    warmup: Optional[int] = None,
    gradient_clip: Optional[float] = 15.0,
    alpha: float = 1e-1,
    one_cycle: bool = False,
):

    chain = []

    # clip gradients
    if gradient_clip is not None:
        chain.append(optax.clip(gradient_clip))

    # choose the optimizer
    if name == "adam":
        chain.append(optax.adam(lr, b1=0.9, b2=0.99, eps=1e-5),)
    else:
        raise ValueError(f"Unrecognized optimizer: {name}")

    # cosine decay learning rate
    if one_cycle:

        one_cycle_cosine_lr = optax.cosine_onecycle_schedule(
            transition_steps=cosine_decay_steps,
            pct_start=0.3,
            peak_value=lr,
            div_factor=25.0,
            final_div_factor=1e4,
        )

        chain.append(optax.scale_by_schedule(one_cycle_cosine_lr))

    elif cosine_decay_steps is not None and cosine_decay_steps > 0:
        cosine_lr = optax.cosine_decay_schedule(
            init_value=1.0, decay_steps=cosine_decay_steps, alpha=alpha
        )
        chain.append(optax.scale_by_schedule(cosine_lr))

    # create optimizer

    return optax.chain(*chain)


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
        default=15.0,
        help="Standardize Input Training Data",
    )
    parser.add_argument(
        "--cosine_decay_steps",
        type=int,
        default=1_000,
        help="Standardize Input Training Data",
    )
    parser.add_argument(
        "--lr_alpha", type=float, default=1e-1, help="Standardize Input Training Data",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Standardize Input Training Data",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=1_000,
        help="Standardize Input Training Data",
    )
    return parser
