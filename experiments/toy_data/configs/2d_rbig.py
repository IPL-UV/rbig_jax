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
from rbig_jax.information.entropy import histogram_entropy
from rbig_jax.models import IterativeGaussianization
from rbig_jax.plots import plot_info_loss, plot_joint, plot_joint_prob
from rbig_jax.transforms.block import get_default_rbig_block
from rbig_jax.transforms.histogram import InitUniHistUniformize
from rbig_jax.transforms.kde import InitKDEUniformize
from rbig_jax.transforms.rotation import InitPCARotation

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
    "--support_extension", type=int, default=10, help="Standardize Input Training Data"
)
parser.add_argument(
    "--alpha", type=float, default=1e-5, help="Standardize Input Training Data"
)
parser.add_argument(
    "--precision", type=int, default=1_000, help="Standardize Input Training Data"
)
parser.add_argument(
    "--nbins", type=str, default="sqrt", help="Standardize Input Training Data"
)
parser.add_argument(
    "--bw", type=float, default=0.1, help="Standardize Input Training Data"
)
parser.add_argument(
    "--eps", type=float, default=1e-5, help="Standardize Input Training Data"
)
parser.add_argument(
    "--n_layers", type=int, default=30, help="Standardize Input Training Data"
)
parser.add_argument(
    "--zero_tol", type=int, default=30, help="Standardize Input Training Data"
)
parser.add_argument(
    "--marginal", type=str, default="histogram", help="Standardize Input Training Data",
)

# ======================
# Logger Parameters
# ======================
parser.add_argument("--wandb-entity", type=str, default="ipl_uv")
parser.add_argument("--wandb-project", type=str, default="rbig_2d_data")
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


if wandb_logger.config.marginal == "histogram":
    support_extension = wandb_logger.config.support_extension
    alpha = wandb_logger.config.alpha
    precision = wandb_logger.config.precision

    if wandb_logger.config.nbins == "sqrt":
        nbins = int(np.sqrt(X.shape[0]))
    else:
        raise ValueError(f"Unrecognized nbins method: {wandb_logger.config.nbins}")

    # initialize histogram transformation
    uni_uniformize = InitUniHistUniformize(
        n_samples=X.shape[0],
        nbins=nbins,
        support_extension=support_extension,
        precision=precision,
        alpha=alpha,
    )
elif wandb_logger.config.marginal == "kde":
    support_extension = wandb_logger.config.support_extension
    bw = wandb_logger.config.bw
    precision = wandb_logger.config.precision

    # initialize histogram transformation
    uni_uniformize = InitKDEUniformize(
        n_samples=X.shape[0],
        support_extension=support_extension,
        precision=precision,
        bw=bw,
    )
else:
    raise ValueError(
        f"Unrecognized marginal transformation: {wandb_logger.config.marginal}"
    )

# initialize PCA Rotation transformation
rot_transform = InitPCARotation()

# model hyperparameters
# histogram transformation parameters
max_layers = 1_000
zero_tolerance = wandb_logger.config.zero_tol
itergauss_model = IterativeGaussianization(
    uni_uniformize=uni_uniformize,
    rot_transform=rot_transform,
    max_layers=max_layers,
    n_features=n_features,
    zero_tolerance=zero_tolerance,
)


# =========================
# TRAINING LOOP
# =========================
# itergauss_model.fit(X)

X_g = itergauss_model.fit_transform(X)


# =========================
# Initialize Figure
# =========================
# plot data
plt.figure()
sns.jointplot(X_g[:, 0], X_g[:, 1], s=5, color="red")
# Log the plot
wandb.log({"trained_latent": wandb.Image(plt)})

# ========================
# GENERATE SAMPLES
# ========================
# generate samples in the latent domain

n_gen_samples = 10_000

X_samples = itergauss_model.sample(n_gen_samples)

# plot data
plt.figure()
sns.jointplot(X_samples[:, 0], X_samples[:, 1], s=5, color="purple")
# Log the plot
wandb.log({"gen_samples": wandb.Image(plt)})


# ========================
# PROBABILITIES
# ========================
XY_input = generate_2d_grid(data=X, buffer=0.1)

X_log_prob = itergauss_model.score_samples(XY_input)


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

# =========================
# FINAL SCORE
# =========================
nll = itergauss_model.score(X)
wandb.log({"nll": float(nll)})

# =========================
# TOTAL CORRELATION
# =========================
TC = itergauss_model.total_correlation()
wandb.log({"Total Correlation": float(TC)})
# =========================
# ENTROPY
# =========================
# marginal entropy
X_mH = jax.vmap(histogram_entropy, in_axes=(1, None, None))(X, 2, 100).sum()

# total entropy
H = X_mH - TC

wandb.log({"Entropy": float(H)})
