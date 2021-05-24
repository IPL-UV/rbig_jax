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
import tensorflow as tf
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
from typing import Optional
from chex import Array
from einops import rearrange

PRNGKey = Array

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


elif wandb_logger.config.dataset == "cifar10":

    image_shape = ImageShape(C=3, H=32, W=32)

    def load_dataset(split: tfds.Split, batch_size: int) -> Iterator[Batch]:
        ((x_train_raw, _), (x_test_raw, _),) = tf.keras.datasets.cifar10.load_data()
        if split == "train":
            ds = tf.data.Dataset.from_tensor_slices(x_train_raw)
        else:
            ds = tf.data.Dataset.from_tensor_slices(x_test_raw)
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=20 * batch_size)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        ds = ds.repeat()
        return iter(tfds.as_numpy(ds))

    def prepare_data(batch: Batch, prng_key: Optional[PRNGKey] = None) -> Array:

        # select image from tfds
        data = batch.astype(jnp.float32)

        # dequantize pixels (training only)
        if prng_key is not None:
            # Dequantize pixel values {0, 1, ..., 255} with uniform noise [0, 1).
            data += jax.random.uniform(prng_key, data.shape).astype(jnp.float32)

        # flatten image data
        data = rearrange(data, "B H W C -> B (H W C)")

        return data / 256.0  # Normalize pixel values from [0, 256) to [0, 1).


else:
    raise ValueError(f"Unrecognized dataset: {wandb_logger.config.dataset}")


KEY = jax.random.PRNGKey(wandb_logger.config.seed)

train_ds = load_dataset(tfds.Split.TRAIN, wandb_logger.config.batch_size)
init_ds = load_dataset(tfds.Split.TRAIN, wandb_logger.config.n_init_samples)
valid_ds = load_dataset(tfds.Split.TEST, wandb_logger.config.val_batch_size)

# demo batch
init_batch = next(init_ds)

# plot batch of images
if wandb_logger.config.dataset == "cifar10":
    fig, ax = plot_image_grid(init_batch)
else:
    fig, ax = plot_image_grid(init_batch["image"])

# ==============================
# %% PREPROCESSING
# ==============================


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
# %% OPTIMIZATION
# ===============================

from rbig_jax.training.parametric import init_optimizer

optimizer = init_optimizer(
    "adam",
    n_epochs=wandb_logger.config.epochs,
    lr=wandb_logger.config.lr,
    cosine_decay_steps=wandb_logger.config.epochs,
    warmup=None,
    gradient_clip=wandb_logger.config.gradient_clip,
    alpha=1e-1,
)


# ===============================
# %% TRAIN STEP
# ===============================
from rbig_jax.training.parametric import GaussFlowTrainer


# initial flow trainer
nf_trainer = GaussFlowTrainer(
    gf_model,
    optimizer,
    n_epochs=wandb_logger.config.epochs,
    prepare_data_fn=prepare_data,
)


# ====================================
# %% TRAINING
# ====================================
import tqdm
from rbig_jax.losses import nll_2_bpd

train_ds = load_dataset(tfds.Split.TRAIN, wandb_logger.config.batch_size)
valid_ds = load_dataset(tfds.Split.TEST, wandb_logger.config.batch_size)


# split the keys into a unique subset
train_rng = jax.random.split(rng, num=wandb_logger.config.epochs)

# create an iterator
train_rng = iter(train_rng)


eval_loss = 0.0
img_shape = X_init.shape[1:]

with tqdm.trange(wandb_logger.config.epochs) as pbar:
    for step in pbar:

        # Train Step
        output = nf_trainer.train_step(next(train_ds), rng=next(train_rng))
        train_loss = output.loss
        train_loss = nll_2_bpd(train_loss, img_shape)
        pbar.set_description(
            f"Train Loss: {train_loss:.4f} | Valid Loss: {eval_loss:.4f}"
        )
        wandb.log({"train_loss": float(train_loss), "training_step": step})

        # Eval Step
        if step % wandb_logger.config.eval_freq == 0:
            output = nf_trainer.validation_step(next(train_ds))
            eval_loss = output.loss
            eval_loss = nll_2_bpd(eval_loss, X_init.shape[1:])
            pbar.set_description(
                f"Train Loss: {train_loss:.4f} | Valid Loss: {eval_loss:.4f}"
            )
            wandb.log({"validation_loss": float(eval_loss), "training_step": step})

            # --------------------
            # Latent Space (Images + Histogram)
            # --------------------
            # forward propagation for data
            X_demo_g = output.model.forward(X_init)

            # plot demo images
            # plot image grid
            fig, ax = plot_image_grid(X_demo_g, image_shape)

            wandb.log({"training_latent_images": wandb.Image(plt)})
            plt.close(fig)

            fig = corner.corner(np.array(X_demo_g[:, :10]), color="red")

            wandb.log({"training_latent_histogram": wandb.Image(plt)})
            plt.close(fig)

            n_gen_samples = 50
            X_samples = output.model.sample(seed=42, n_samples=n_gen_samples)

            # plot
            fig, ax = plot_image_grid(X_samples, image_shape)

            wandb.log({"training_generated_images": wandb.Image(plt)})
            plt.close(fig)


gf_model = output.model

# ====================================
# PLOTTING
# ====================================

# --------------------
# Loss Function
# --------------------
print("Plotting Loss Function...")
fig, ax = plt.subplots()
ax.plot(
    nf_trainer.train_epoch, nf_trainer.train_loss, label="Training Loss", color="blue"
)
ax.plot(
    nf_trainer.valid_epoch,
    nf_trainer.valid_loss,
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
