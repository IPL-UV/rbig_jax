# jax packages
import itertools
import os
import sys
from argparse import ArgumentParser
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np

# plot methods
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import corner

sns.reset_defaults()
sns.set_context(context="poster", font_scale=0.7)

# logging
import tqdm
import wandb
from celluloid import Camera
from jax import device_put, random
from jax.config import config
from pyprojroot import here
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from wandb.sdk import wandb_config

# library functions
from rbig_jax.models.gaussflow import GaussianizationFlow, init_default_gf_model


# spyder up to find the root
root = here(project_files=[".here"])

# append to path
sys.path.append(str(here()))


# import chex
config.update("jax_enable_x64", False)


sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)

# ==========================
# PARAMETERS
# ==========================

parser = ArgumentParser(
    description="2D Data Demo with Iterative Gaussianization method"
)

# ======================
# Dataset
# ======================
from rbig_jax.data import add_dataset_args

parser = add_dataset_args(parser)


# ======================
# Preprocessing
# ======================
parser.add_argument(
    "--standardize", type=bool, default=True, help="Standardize Input Training Data"
)

# ======================
# Model
# ======================
from rbig_jax.models.gaussflow import add_gf_model_args

parser = add_gf_model_args(parser)

# ======================
# Model Training
# ======================
from rbig_jax.training.parametric import add_gf_train_args

parser = add_gf_train_args(parser)

# ======================
# Logger Parameters
# ======================
parser.add_argument("--wandb-entity", type=str, default="ipl_uv")
parser.add_argument("--wandb-project", type=str, default="gf_2d_data")
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

# ==========================
# INITIALIZE LOGGER
# ==========================

wandb_logger = wandb.init(project=args.wandb_project, entity=args.wandb_entity)
wandb_logger.config.update(args)
seed = wandb_logger.config.seed
# ==========================
#  LOAD DATA
# ==========================

# get data
n_samples = wandb_logger.config.n_samples
n_features = 2

if wandb_logger.config.dataset in ["noisysine"]:
    from rbig_jax.data import NoisySineDataset as PlaneDataset

elif wandb_logger.config.dataset in ["s_curve"]:
    from rbig_jax.data import SCurveDataset as PlaneDataset

elif wandb_logger.config.dataset in ["moons"]:
    from rbig_jax.data import MoonsDataset as PlaneDataset

elif wandb_logger.config.dataset in ["swiss_roll"]:
    from rbig_jax.data import SwissRollDataset as PlaneDataset

elif wandb_logger.config.dataset in ["blobs"]:
    from rbig_jax.data import BlobsDataset as PlaneDataset
elif wandb_logger.config.dataset in ["checkerboard"]:
    from rbig_jax.data import CheckBoard as PlaneDataset
else:
    raise ValueError(f"Unrecognized dataset: {wandb_logger.config.dataset}")

# initialize dataset
ds_train = PlaneDataset(n_samples=args.n_train, noise=args.noise, seed=args.seed)
ds_valid = PlaneDataset(n_samples=args.n_valid, noise=args.noise, seed=args.seed + 1)
ds_plot = PlaneDataset(n_samples=1_000_000, noise=args.noise, seed=args.seed + 2)

# ==========================
# Train-Test Split
# ==========================
from torch.utils.data import DataLoader

# initialize dataloader
batch_size = 256
shuffle = True

train_dl = DataLoader(
    ds_train, batch_size=batch_size, shuffle=shuffle, collate_fn=None, num_workers=0
)
valid_dl = DataLoader(
    ds_valid, batch_size=batch_size, shuffle=False, collate_fn=None, num_workers=0
)

# ==========================
#  PLOTTING
# ==========================

# plot data

fig = corner.corner(ds_train[:], color="blue", hist_bin_factor=2)
wandb.log({"original_data": wandb.Image(plt)})


# ==========================
#  BUILD MODEL
# ==========================
if args.n_init_samples < len(ds_train):
    X_init = ds_train[: args.n_init_samples]
else:
    X_init = ds_train[:]


# initialize Model
gf_model = init_default_gf_model(
    shape=X_init.shape[1:],
    X=X_init,
    n_blocks=args.n_blocks,
    mixture=args.mixture,
    n_components=args.n_components,
    init_mixcdf=args.init_mixcdf,
    inverse_cdf=args.inverse_cdf,
    init_rotation=args.init_rotation,
    n_reflections=args.n_reflections,
    plot_layers=False,
    plot_blocks=False,
)


# ==========================
#  PLOTTING
# ==========================
Z = gf_model.forward(X_init)

fig = corner.corner(Z, color="red", hist_bin_factor=2)
wandb.log({"initial_latent": wandb.Image(plt)})

# ==========================
#  TRAINING
# ==========================

from jax.experimental import optimizers


if args.optimizer == "adam":

    optimizer = optimizers.adam(step_size=args.lr)
elif args.optimizer == "sgd":
    optimizer = optimizers.adam(step_size=args.lr)
else:
    raise ValueError(f"Unrecognized optimizer: {args.optimizer}")

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


train_op = jax.jit(train_op)

# ================================
# TRAINING
# ================================
import jax.numpy as jnp

train_losses = list()
valid_losses = list()
itercount = itertools.count()
train_batch_loss = 0.0
valid_batch_loss = 0.0
interval = 5
pbar = tqdm.trange(args.epochs)

with pbar:
    for i_epoch in pbar:

        # Train
        avg_loss = []

        for ix in train_dl:

            # cast to jax array
            ix = jnp.array(ix, dtype=jnp.float32)

            # compute loss
            loss, opt_state = train_op(next(itercount), opt_state, ix,)

            # append batch
            wandb.log({"train_loss_batch": float(loss), "epoch": i_epoch})
            avg_loss.append(float(loss))

        # average loss
        train_batch_loss = jnp.mean(jnp.stack(avg_loss))
        wandb.log({"train_loss": float(train_batch_loss), "epoch": i_epoch})

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
                wandb.log({"valid_loss_batch": float(loss), "epoch": i_epoch})
                avg_loss.append(float(loss))

            # average loss
            valid_batch_loss = jnp.mean(jnp.stack(avg_loss))
            wandb.log({"valid_loss": float(valid_batch_loss), "epoch": i_epoch})

            valid_losses.append(np.array(valid_batch_loss))

            pbar.set_postfix(
                {
                    "Train Loss": f"{train_batch_loss:.4f}",
                    "Valid Loss": f"{valid_batch_loss:.4f}",
                }
            )

        else:
            continue

        if i_epoch % interval == 0:
            X_trans = []
            for ix in train_dl:

                # cast to jax array
                ix = jnp.array(ix, dtype=jnp.float32)

                ix = final_params.forward(ix)

                X_trans.append(ix)

            X_trans = jnp.concatenate(X_trans, axis=0)
            fig = corner.corner(np.array(X_trans), color="Red")
            wandb.log({"training_latent_space": wandb.Image(plt), "epoch": i_epoch})


final_bijector = get_params(opt_state)


# ==========================
#  Score
# ==========================
nll_loss = final_bijector.score(ds_train[:])

wandb.log({"nll_loss": np.array(nll_loss)})


# ==========================
#  PLOTTING
# ==========================
train_losses = jnp.stack(train_losses)
if valid_dl is not None:
    valid_losses = jnp.stack(valid_losses)
else:
    valid_losses = None
losses = {"train": train_losses, "valid": valid_losses}


fig, ax = plt.subplots()
ax.plot(losses["train"], label="Training Loss", color="blue")
ax.plot(losses["valid"], label="Validation Loss", color="orange")
ax.set(xlabel="Iterations", ylabel="Negative Log-Likelihood")
plt.legend()
plt.tight_layout()
wandb.log({"final_losses": wandb.Image(plt)})

# =========================
# PLOTS
# =========================

# LATENT SPACE
X_trans = final_bijector.forward(ds_train[:])

fig = corner.corner(X_trans, color="Red")
wandb.log({"final_latent_space": wandb.Image(plt)})

# INVERSE TRANSFORM
X_approx = final_bijector.inverse(X_trans)

fig = corner.corner(X_approx, color="green")
wandb.log({"inverse_space": wandb.Image(plt)})

# LOG PROBABILITY
from rbig_jax.data import generate_2d_grid

cmap = cm.magma  # "Reds"
X_plot = ds_plot[:]

fig, ax = plt.subplots(figsize=(7, 5))
h = ax.hist2d(
    X_plot[:, 0], X_plot[:, 1], bins=512, cmap=cmap, density=True, vmin=0.0, vmax=1.0
)
ax.set(
    xlim=[X_plot[:, 0].min(), X_plot[:, 0].max()],
    ylim=[X_plot[:, 1].min(), X_plot[:, 1].max()],
    xticklabels="",
    yticklabels="",
)
plt.tight_layout()
wandb.log({"original_density": wandb.Image(plt)})


# generate grid points
xyinput = generate_2d_grid(ds_train[:], 500, buffer=0.2)

# calculate log probability
X_log_prob = final_bijector.score_samples(xyinput)

X_plot = ds_plot[:]

# Estimated Density

probs = jnp.exp(X_log_prob)

fig, ax = plt.subplots(figsize=(7, 5))
h = ax.scatter(
    xyinput[:, 0], xyinput[:, 1], s=1, c=probs, cmap=cmap, vmin=0.0, vmax=1.0
)
ax.set(
    xlim=[xyinput[:, 0].min(), xyinput[:, 0].max()],
    ylim=[xyinput[:, 1].min(), xyinput[:, 1].max()],
    xticklabels="",
    yticklabels="",
)
plt.tight_layout()
wandb.log({"estimated_density": wandb.Image(plt)})
# =======================
# Save Model
# =======================
import joblib

model_save_name = os.path.join(wandb.run.dir, "trained_model.pckl")
joblib.dump(final_bijector, model_save_name)
wandb.save(model_save_name)
