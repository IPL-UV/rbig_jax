# jax packages
import itertools
import os
import sys
from argparse import ArgumentParser
from functools import partial

import chex
import jax
import jax.numpy as np
# plot methods
import matplotlib.pyplot as plt
import numpy as onp
import objax
import seaborn as sns
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
from rbig_jax.data import generate_2d_grid, get_classic
from rbig_jax.models.gaussflow import GaussianizationFlow
from rbig_jax.plots import plot_info_loss, plot_joint, plot_joint_prob
from rbig_jax.transforms.base import CompositeTransform
from rbig_jax.transforms.inversecdf import InverseGaussCDF
from rbig_jax.transforms.linear import HouseHolder
from rbig_jax.transforms.logit import Logit
from rbig_jax.transforms.mixture import MixtureGaussianCDF, MixtureLogisticCDF

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
parser.add_argument(
    "--seed", type=int, default=123, help="number of data points for training",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="classic",
    help="Dataset to be used for visualization",
)
parser.add_argument(
    "--n-samples", type=int, default=10_000, help="number of data points for training",
)


# ======================
# Preprocessing
# ======================
parser.add_argument(
    "--standardize", type=bool, default=True, help="Standardize Input Training Data"
)

# ======================
# Model
# ======================
parser.add_argument(
    "--n-components", type=int, default=20, help="Standardize Input Training Data"
)
parser.add_argument(
    "--n-reflections", type=int, default=2, help="Standardize Input Training Data"
)
parser.add_argument(
    "--temperature", type=bool, default=False, help="Standardize Input Training Data"
)
parser.add_argument(
    "--n-layers", type=int, default=4, help="Standardize Input Training Data"
)
parser.add_argument(
    "--quantile", type=str, default="logit", help="Standardize Input Training Data"
)
parser.add_argument(
    "--mixture", type=str, default="gaussian", help="Standardize Input Training Data"
)

# ======================
# Model Training
# ======================
parser.add_argument("--batchsize", type=int, default=128, help="Number of batches")
parser.add_argument("--epochs", type=int, default=500, help="Number of batches")
parser.add_argument(
    "--learning-rate", type=float, default=0.01, help="Number of batches"
)
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

if wandb_logger.config.dataset in ["classic"]:
    data = get_classic(n_samples)
elif wandb_logger.config.dataset in ["helix"]:
    noise = 0.5
    wandb_logger.config.update({"noise": noise})
    data, _ = datasets.make_swiss_roll(
        n_samples=n_samples, noise=noise, random_state=seed
    )
    data = data[:, [0, 2]]

elif wandb_logger.config.dataset in ["moons"]:
    noise = 0.05
    wandb_logger.config.update({"noise": noise})
    data, _ = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=seed)

elif wandb_logger.config.dataset in ["swiss_roll"]:
    noise = 0.1
    wandb_logger.config.update({"noise": noise})
    data, _ = datasets.make_s_curve(n_samples=n_samples, noise=noise, random_state=seed)
    data = data[:, [0, 2]]

elif wandb_logger.config.dataset in ["blobs"]:
    data, _ = datasets.make_blobs(n_samples=n_samples, random_state=seed)
else:
    raise ValueError(f"Unrecognized dataset: {wandb_logger.config.dataset}")

# ==========================
#  PREPROCESSING
# ==========================

if wandb_logger.config.standardize is True:
    data = StandardScaler().fit_transform(data)

# ==========================
#  PLOTTING
# ==========================

# plot data
plt.figure()
sns.jointplot(data[:, 0], data[:, 1], s=5, color="blue")
# Log the plot
wandb.log({"original_data": wandb.Image(plt)})

# ==========================
#  PLOTTING
# ==========================
X = np.array(data, dtype=np.float32)
X = jax.device_put(X)


# ==========================
#  BUILD MODEL
# ==========================
# model hyperparameters
n_components = wandb_logger.config.n_components
n_reflections = wandb_logger.config.n_reflections
generator = objax.random.Generator(123)
learn_temperature = wandb_logger.config.temperature
n_features = data.shape[1]


n_layers = wandb_logger.config.n_layers

transforms = []

for _ in range(n_layers):
    # mixture gaussian cdf distribution, f:[-inf,inf] -> [0,1]
    if wandb_logger.config.mixture == "gaussian":
        transforms.append(
            MixtureGaussianCDF(n_features=n_features, n_components=n_components)
        )
    elif wandb_logger.config.mixture == "logistic":
        transforms.append(
            MixtureLogisticCDF(n_features=n_features, n_components=n_components)
        )
    else:
        raise ValueError(f"Unrecognized mixture layer: {wandb_logger.config.mixture}")

    # Logit quantile function, f:[0,1] -> [-inf,inf]
    if wandb_logger.config.quantile == "logit":
        transforms.append(Logit(learn_temperature=learn_temperature))
    elif wandb_logger.config.quantile == "igausscdf":
        transforms.append(InverseGaussCDF())
    else:
        raise ValueError(
            f"Unrecognized quantile transform: {wandb_logger.config.quantile}"
        )

    # orthogonal rotation layer
    transforms.append(
        HouseHolder(
            n_features=n_features, n_reflections=n_reflections, generator=generator
        )
    )

# compose all transformations into a single chain
transform = CompositeTransform(transforms)

# initialize base distribution
base_dist = jax.scipy.stats.norm

# initialize Model
gf_model = GaussianizationFlow(
    n_features=n_features, bijections=transform, base_dist=base_dist
)


# ==========================
#  PLOTTING
# ==========================
Z, _ = gf_model(X)

# plot data
plt.figure()
sns.jointplot(Z[:, 0], Z[:, 1], s=5, color="red")
# Log the plot
wandb.log({"initial_latent": wandb.Image(plt)})

# ==========================
#  Loss Function
# ==========================
@objax.Function.with_vars(gf_model.vars())
def nll_loss(x):
    return gf_model.score(x)


# =========================
# Optimizer
# =========================
# define the optimizer
opt = objax.optimizer.Adam(gf_model.vars())

# get grad values
gv = objax.GradValues(nll_loss, gf_model.vars())
lr = wandb_logger.config.learning_rate
epochs = wandb_logger.config.epochs
batchsize = wandb_logger.config.batchsize

# define the training operation
@objax.Function.with_vars(gf_model.vars() + opt.vars())
def train_op(x):
    g, v = gv(x)  # returns gradients, loss
    opt(lr, g)
    return v


# This line is optional: it is compiling the code to make it faster.
train_op = objax.Jit(train_op)

# initialize parameters
key = random.PRNGKey(123)
itercount = itertools.count()
permute_rng, rng = random.split(key)

losses = list()

# =========================
# Initialize Figure
# =========================
# initialize figure
fig = plt.figure()
camera = Camera(fig)
# make predictions
Z = gf_model.transform(X)
plt.scatter(Z[:, 0], Z[:, 1], s=1, color="Red")
camera.snap()

# =========================
# TRAINING LOOP
# =========================

pbar = tqdm.trange(epochs)

with pbar:
    for i in pbar:

        # batch processing
        permute_rng, rng = random.split(rng)

        # randomly shuffle the data
        train_data = random.permutation(permute_rng, X)

        # Train
        avg_loss = []

        for batch_index in range(0, n_samples, batchsize):
            # compute loss
            loss = float(train_op(train_data[batch_index : batch_index + batchsize])[0])
            # append batch
            avg_loss.append(loss)
        # average loss
        batch_loss = np.mean(np.stack(avg_loss))

        wandb.log({"nll_loss": float(batch_loss)})

        # Log losses
        losses.append(batch_loss)
        pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

        if (i + 1) % 20 == 0 or i < 20:

            # make predictions
            Z = gf_model.transform(X)
            plt.scatter(Z[:, 0], Z[:, 1], s=1, color="Red")
            camera.snap()

animation = camera.animate(250)
# HTML(animation.to_html5_video())
animation.save("./training.gif")
wandb.log({"training_latent": wandb.Video("./training.gif", fps=1, format="gif")})

# =========================
# FINAL SCORE
# =========================
nll = gf_model.score(X)

wandb.log({"nll": float(nll)})

# =========================
# PLOTTING
# =========================

# forward transformation
Z, logabsdet = gf_model(X)

# plot data
plt.figure()
sns.jointplot(Z[:, 0], Z[:, 1], s=5, color="red")
# Log the plot
wandb.log({"final_latent": wandb.Image(plt)})


# ========================
# GENERATE SAMPLES
# ========================
# generate samples in the latent domain

# inverse transformation
X_samples = gf_model.sample(10_000)

# plot data
plt.figure()
sns.jointplot(X_samples[:, 0], X_samples[:, 1], s=5, color="purple")
# Log the plot
wandb.log({"gen_samples": wandb.Image(plt)})


# ========================
# PROBABILITIES
# ========================
XY_input = generate_2d_grid(data=X, buffer=0.1)

X_log_prob = gf_model.score_samples(XY_input)


cmap = "Reds"
probs = np.exp(X_log_prob)
# probs = np.clip(probs, 0.0, 1.0)
title = "Probability"

fig, ax = plt.subplots()
h = ax.scatter(
    XY_input[:, 0], XY_input[:, 1], s=1, c=probs, cmap=cmap, vmin=0.0, vmax=1.0
)
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(h)
ax.set_title(title)
plt.tight_layout()
wandb.log({"probs": wandb.Image(plt)})

# ========================
# DEMO LAYER TRANSFORMS
# ========================

# initialize figure
fig, ax = plt.subplots()
camera = Camera(fig)

outputs = X


for ilayer, itransform in enumerate(gf_model.bijections._transforms):
    outputs = itransform.transform(outputs)

    if (ilayer + 1) % 3 == 0:
        # make predictions
        ax.scatter(outputs[:, 0], outputs[:, 1], s=1, color="Red")
        ax.set_xlim([outputs[:, 0].min() - 0.1, outputs[:, 0].max() + 0.1])
        ax.set_ylim([outputs[:, 1].min() - 0.1, outputs[:, 1].max() + 0.1])
        ax.text(
            0.4,
            1.05,
            f"Layer: {int((ilayer+1)/3)}",
            transform=ax.transAxes,
            fontsize=20,
        )
        camera.snap()

animation = camera.animate(1_500)
# HTML(animation.to_html5_video())
animation.save("./layer_transforms.gif")
wandb.log({"layers": wandb.Video("./layer_transforms.gif", fps=1, format="gif")})
