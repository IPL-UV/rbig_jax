import sys, os
from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".here"])

# append to path
sys.path.append(str(root))

# jax packages
import jax
import jax.numpy as jnp
from jax.config import config

# import chex
config.update("jax_enable_x64", False)

from argparse import ArgumentParser
import chex
import numpy as np
from functools import partial


# Plot utilities
from rbig_jax.custom_types import ImageShape
from rbig_jax.plots import plot_image_grid
from rbig_jax.training.parametric import add_gf_train_args
from rbig_jax.models.gaussflow import add_gf_model_args


# logging
import tqdm
import wandb

# plot methods
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import corner

sns.reset_defaults()
sns.set_context(context="poster", font_scale=0.7)


# ==========================
# PARAMETERS
# ==========================

parser = ArgumentParser(
    description="2D Data Demo with Iterative Gaussianization method"
)

# Dataset
parser.add_argument(
    "--seed", type=int, default=123, help="number of data points for training",
)
parser.add_argument(
    "--dataset", type=str, default="mnist", help="number of data points for training",
)

# Model
parser = add_gf_model_args(parser)
# Training
parser = add_gf_train_args(parser)

# ======================
# Logger Parameters
# ======================
parser.add_argument("--wandb-entity", type=str, default="ipl_uv")
parser.add_argument("--wandb-project", type=str, default="gf_mnist_naive")
# =====================
# Testing
# =====================
parser.add_argument(
    "-sm",
    "--smoke-test",
    action="store_true",
    help="to do a smoke test without logging",
)

args = parser.parse_args()
# change this so we don't bug wandb with our BS
if args.smoke_test:
    os.environ["WANDB_MODE"] = "dryrun"
    args.epochs = 1
    args.n_samples = 1_000
    args.n_blocks = 1
    args.n_init_samples = 100


# ============================
# %% LOGGING
# ============================
wandb_logger = wandb.init(project=args.wandb_project, entity=args.wandb_entity)
wandb_logger.config.update(args)

# ============================
# %% DATA
# ============================
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Iterator, Mapping
import numpy as np

Batch = Mapping[str, np.ndarray]

if wandb_logger.config.dataset == "mnist":

    image_shape = ImageShape(C=1, H=28, W=28)

    def load_dataset(split: tfds.Split, batch_size: int) -> Iterator[Batch]:
        ds = tfds.load("mnist", split=split, shuffle_files=True)
        ds = ds.shuffle(buffer_size=10 * batch_size)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        ds = ds.repeat()
        return iter(tfds.as_numpy(ds))


elif wandb_logger.config.dataset == "cifar10":

    image_shape = ImageShape(C=3, H=32, W=32)

    def load_dataset(split: tfds.Split, batch_size: int) -> Iterator[Batch]:
        ds = tfds.load("cifar10", split=split, shuffle_files=True)
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=20 * batch_size)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        ds = ds.repeat()
        return iter(tfds.as_numpy(ds))


else:
    raise ValueError(f"Unrecognized dataset: {wandb_logger.config.dataset}")


KEY = jax.random.PRNGKey(wandb_logger.config.seed)

train_ds = load_dataset(tfds.Split.TRAIN, wandb_logger.config.batch_size)
init_ds = load_dataset(tfds.Split.TRAIN, wandb_logger.config.n_init_samples)
valid_ds = load_dataset(tfds.Split.TEST, wandb_logger.config.val_batch_size)

# demo batch
init_batch = next(init_ds)

# plot batch of images
fig, ax = plot_image_grid(init_batch["image"])

# ==============================
# %% PREPROCESSING
# ==============================
from typing import Optional
from chex import Array
from einops import rearrange

PRNGKey = Array


def prepare_data(batch: Batch, prng_key: Optional[PRNGKey] = None) -> Array:

    # select image from tfds
    data = batch["image"].astype(jnp.float32)

    # dequantize pixels (training only)
    if prng_key is not None:
        # Dequantize pixel values {0, 1, ..., 255} with uniform noise [0, 1).
        data += jax.random.uniform(prng_key, data.shape).astype(jnp.float32)

    # flatten image data
    data = rearrange(data, "B H W C -> B (H W C)")

    return data / 256.0  # Normalize pixel values from [0, 256) to [0, 1).


# Demo
# create key
rng, prng_key = jax.random.split(KEY, num=2)

# prep the data
demo_data_prepped = prepare_data(init_batch, prng_key=prng_key)

# plot image grid

fig, ax = plot_image_grid(demo_data_prepped, image_shape)
wandb.log({"initial_images": wandb.Image(plt)})
plt.close(fig)
# ==============================
# %% MODEL
# ==============================
from rbig_jax.models.gaussflow import add_gf_model_args, init_gf_spline_model
from rbig_jax.models.gaussflow import init_default_gf_model

# initialization data
init_ds = load_dataset(tfds.Split.TRAIN, wandb_logger.config.n_init_samples)
init_ds = next(init_ds)
init_data_prepped = prepare_data(init_ds, prng_key=prng_key)
X_init = np.array(init_data_prepped, dtype=np.float64)


if wandb_logger.config.model == "rqsplines":

    # init model
    gf_model = init_gf_spline_model(
        shape=X_init.shape[1:],
        X=X_init,
        n_blocks=wandb_logger.config.n_blocks,
        n_bins=wandb_logger.config.n_bins,
        range_min=wandb_logger.config.range_min,
        range_max=wandb_logger.config.range_max,
        boundary_slopes=wandb_logger.config.boundary_slopes,
        identity_init=wandb_logger.config.identity_init,
        init_rotation=wandb_logger.config.init_rotation,
        n_reflections=wandb_logger.config.n_reflections,
    )

elif wandb_logger.config.model == "mixture":

    # init model
    gf_model = init_default_gf_model(
        shape=X_init.shape[1:],
        X=X_init,
        n_blocks=wandb_logger.config.n_blocks,
        init_mixcdf=wandb_logger.config.init_mixcdf,
        init_rotation=wandb_logger.config.init_rotation,
        inverse_cdf=wandb_logger.config.inverse_cdf,
        mixture=wandb_logger.config.mixture,
        n_reflections=wandb_logger.config.n_reflections,
        n_components=wandb_logger.config.n_components,
    )
else:
    raise ValueError(f"Unrecognzied model: {wandb_logger.config.model}")

# Demo - Forwad Propagation
# forward propagation for data
X_demo_g = gf_model.forward(demo_data_prepped)


try:
    fig, ax = plot_image_grid(X_demo_g, image_shape)
    wandb.log({"initial_latent_images": wandb.Image(plt)})
    plt.close(fig)
    fig = corner.corner(np.array(X_demo_g[:, :10]), color="red")
    wandb.log({"initial_latent_histogram": wandb.Image(plt)})
except ValueError:
    pass
finally:
    plt.close(fig)

# Demo - Inverse Propagation
X_demo_approx = gf_model.inverse(X_demo_g[:50])

# plot image grid
try:
    plot_image_grid(X_demo_approx, image_shape)
    wandb.log({"initial_inverse_images": wandb.Image(plt)})
except ValueError:
    pass
finally:
    plt.close(fig)

# ===============================
# %% LOSS FUNCTION
# ===============================
from chex import dataclass


def loss_fn(model: dataclass, prng_key: PRNGKey, batch: Batch) -> Array:

    # prepare data
    data = prepare_data(batch, prng_key)

    # negative log likelihood loss
    log_px = model.score_samples(data)

    # calculate bits per dimension
    bpd = -log_px * jnp.log2(jnp.exp(1)) / data.shape[1]

    return jnp.mean(bpd)


@jax.jit
def eval_fn(model: dataclass, batch: Batch) -> Array:

    # prepare data
    data = prepare_data(batch)

    # negative log likelihood loss
    log_px = model.score_samples(data)

    # calculate bits per dimension
    bpd = -log_px * jnp.log2(jnp.exp(1)) / data.shape[1]

    return jnp.mean(bpd)


# initial
train_batch = next(train_ds)
nll_loss = loss_fn(gf_model, prng_key, train_batch)
print(f"Initial NLL Loss (Train): {nll_loss:.4f}")

valid_batch = next(valid_ds)
nll_loss_val = eval_fn(gf_model, valid_batch)

print(f"Initial NLL Loss (Valid): {nll_loss_val:.4f}")

# ===============================
# %% OPTIMIZATION
# ===============================
import optax

# TODO: do an epoch/batchsize calculation for the learning decay

if wandb_logger.config.optimizer == "adam":
    b1 = 0.9
    b2 = 0.99
    eps = 1e-5
    # intialize optimizer
    optimizer = optax.chain(
        optax.clip(wandb_logger.config.gradient_clip),
        optax.adam(wandb_logger.config.lr, b1=b1, b2=b2, eps=eps),
    )

# intialize optimizer state
opt_state = optimizer.init(gf_model)

# ===============================
# %% TRAIN STEP
# ===============================
from typing import Tuple, Any

OptState = Any


@jax.jit
def update(
    params: dataclass, prng_key: PRNGKey, opt_state: OptState, batch: Batch
) -> Tuple[dataclass, OptState]:
    """Single SGD update step."""
    # calculate the loss AND the gradients
    loss, grads = jax.value_and_grad(loss_fn)(params, prng_key, batch)

    # update the gradients
    updates, new_opt_state = optimizer.update(grads, opt_state)

    # update the parameters
    new_params = optax.apply_updates(params, updates)

    # return loss AND new opt_state
    return new_params, new_opt_state, loss


# ====================================
# %% TRAINING
# ====================================
# split the keys into a unique subset
train_rng = jax.random.split(rng, num=wandb_logger.config.epochs)

# create an iterator
train_rng = iter(train_rng)

import tqdm

metrics = {
    "train_step": list(),
    "train_loss": list(),
    "valid_step": list(),
    "valid_loss": list(),
}
train_ds = load_dataset(tfds.Split.TRAIN, wandb_logger.config.batch_size)
valid_ds = load_dataset(tfds.Split.TEST, wandb_logger.config.batch_size)

eval_loss = 0.0
with tqdm.trange(wandb_logger.config.epochs) as pbar:
    for step in pbar:
        gf_model, opt_state, loss = update(
            gf_model, next(train_rng), opt_state, next(train_ds)
        )

        pbar.set_description(f"Train Loss: {loss:.4f} | Valid Loss: {eval_loss:.4f}")
        wandb.log({"train_loss": float(loss), "training_step": step})
        metrics["train_step"].append(step)
        metrics["train_loss"].append(loss)

        if step % wandb_logger.config.eval_freq == 0:
            eval_loss = eval_fn(gf_model, next(valid_ds))

            pbar.set_description(
                f"Train Loss: {loss:.4f} | Valid Loss: {eval_loss:.4f}"
            )
            wandb.log({"validation_loss": float(eval_loss), "training_step": step})
            metrics["valid_step"].append(step)
            metrics["valid_loss"].append(eval_loss)

            # --------------------
            # Latent Space (Images + Histogram)
            # --------------------
            # forward propagation for data
            X_demo_g = gf_model.forward(X_init)

            # plot demo images
            # plot image grid
            fig, ax = plot_image_grid(X_demo_g, image_shape)

            wandb.log({"training_latent_images": wandb.Image(plt)})
            plt.close(fig)

            fig = corner.corner(np.array(X_demo_g[:, :10]), color="red")

            wandb.log({"training_latent_histogram": wandb.Image(plt)})
            plt.close(fig)

            n_gen_samples = 50
            X_samples = gf_model.sample(seed=42, n_samples=n_gen_samples)

            # plot
            fig, ax = plot_image_grid(X_samples, image_shape)

            wandb.log({"training_generated_images": wandb.Image(plt)})
            plt.close(fig)


# ====================================
# PLOTTING
# ====================================

# --------------------
# Loss Function
# --------------------
print("Plotting Loss Function...")
fig, ax = plt.subplots()
ax.plot(
    metrics["train_step"], metrics["train_loss"], label="Training Loss", color="blue"
)
ax.plot(
    metrics["valid_step"],
    metrics["valid_loss"],
    label="Validation Loss",
    color="orange",
)
ax.set(
    xlabel="Iterations", ylabel="Negative Log-Likelihood",
)
plt.legend()
plt.tight_layout()
wandb.log({"final_losses": wandb.Image(plt)})
plt.close(fig)

# --------------------
# Latent Space (Images + Histogram)
# --------------------
print("Plotting Forward Latent Space...")
# forward propagation for data
X_demo_g = gf_model.forward(X_init)

# plot demo images
# plot image grid
fig, ax = plot_image_grid(X_demo_g, image_shape)

wandb.log({"final_latent_images": wandb.Image(plt)})

fig = corner.corner(np.array(X_demo_g[:, :10]), color="red")
wandb.log({"final_latent_histogram": wandb.Image(plt)})
plt.close(fig)

# --------------------
# Inverse Function
# --------------------
print("Plotting Inverse Latent Space...")
# forward propagation for data
X_demo_approx = gf_model.inverse(X_demo_g)

# plot demo images
# plot image grid
fig, ax = plot_image_grid(X_demo_approx, image_shape)
# plot image grid

wandb.log({"inverse_transform_images": wandb.Image(plt)})
plt.close(fig)

# --------------------
# Sampling
# --------------------
print("Generating Samples...")
n_gen_samples = 50
X_samples = gf_model.sample(seed=42, n_samples=n_gen_samples)

# plot
print("Plotting Generated Samples...")
fig, ax = plot_image_grid(X_samples, image_shape)

wandb.log({"generated_samples_images": wandb.Image(plt)})
plt.close(fig)

# --------------------
# Plot Each Layer
# --------------------

# print("Plotting Each Layer...")
# X_g = X_init.copy()

# fig = corner.corner(X_g, color="purple")
# fig.suptitle("Initial")
# plt.show()

# stopping = ""

# for ilayer, ibijector in enumerate(gf_model.bijectors):

#     X_g = ibijector.forward(X_g)

#     if ibijector.name == "HouseHolder":
#         fig = corner.corner(np.array(X_g[:, :10]), color="purple")
#         wandb.log({f"layer_{ilayer}_histogram": wandb.Image(plt)})
#         plt.close(fig)


# ====================================
# SAVING
# ====================================
print("Saving Model...")
import joblib

model_save_name = os.path.join(wandb.run.dir, "gf_model.pckl")
joblib.dump(gf_model, model_save_name)
wandb.save(model_save_name)

print("Done!!")
