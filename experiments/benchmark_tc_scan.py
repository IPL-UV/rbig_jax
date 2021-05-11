# JAX SETTINGS
import time as time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as np
# plot methods
# Plot Functions
import matplotlib.pyplot as plt
import numpy as onp
import seaborn as sns
import tqdm
import wandb
from jax.config import config
from scipy.stats import beta

from rbig_jax.data import get_classic
from rbig_jax.information.entropy import histogram_entropy
from rbig_jax.information.total_corr import rbig_total_correlation
from rbig_jax.plots import plot_info_loss, plot_joint
from rbig_jax.transforms.histogram import histogram_transform

config.update("jax_enable_x64", True)


sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)


def main():
    # SETUP LOGGING
    wandb.init(project="benchmark-tc", entity="emanjohnson91")
    wandb.config.dataset = "random"
    wandb.config.implementation = "rbigjax"
    wandb.config.support_extension = 10
    wandb.config.precision = 100
    wandb.config.alpha = 1e-5
    wandb.config.n_layers = 100
    wandb.config.seed = 123

    rng = onp.random.RandomState(wandb.config.seed)

    # ===================================
    # EXPERIMENT II - DIMENSIONS VS TIME
    # ===================================

    times = list()
    dimensions = [2, 5, 10, 25, 50, 100, 1_000, 10_000]
    n_layers = 100
    n_samples = 10_000
    tc_true, tc_est = [], []

    with tqdm.tqdm(dimensions, desc="Dimensions") as pbar:
        for idim in pbar:

            fake_data = rng.randn(n_samples, idim)

            # Generate random Data
            A = rng.rand(idim, idim)

            fake_data = fake_data @ A
            # covariance matrix
            C = A.T @ A
            vv = onp.diag(C)

            tc = onp.log(onp.sqrt(vv)).sum() - 0.5 * onp.log(onp.linalg.det(C))

            wandb.log({"dim_tc_true": tc}, step=idim)
            tc_true.append(tc)
            t0 = time.time()

            # define marginal entropy function
            nbins = int(onp.ceil(onp.sqrt(fake_data.shape[0])))
            entropy_f = jax.partial(histogram_entropy, nbins=nbins, base=2)

            # define marginal uniformization function
            hist_transform_f = jax.partial(histogram_transform, nbins=nbins)

            # find the total correlation
            X_trans, loss = rbig_total_correlation(
                np.array(fake_data).block_until_ready(),
                marginal_uni=hist_transform_f,
                uni_entropy=entropy_f,
                n_iterations=n_layers,
            )
            tc = onp.array(np.sum(loss) * np.log(2))

            t1 = time.time() - t0
            wandb.log({"time": t1}, step=idim)
            wandb.log({"dim_tc_est": tc}, step=idim)
            tc_est.append(tc)
            times.append(t1)

    fig, ax = plt.subplots()
    ax.plot(dimensions, times)
    ax.set(xlabel="Dimensions", ylabel="Timing (seconds)")
    plt.yscale("log")
    plt.tight_layout()
    wandb.log({f"Dimensions": [wandb.Image(plt)]})

    fig, ax = plt.subplots()
    ax.plot(dimensions, tc_est, label="Estimated")
    ax.plot(dimensions, tc_true, label="True")
    ax.set(xlabel="Dimensions", ylabel="Timing (seconds)")
    plt.tight_layout()
    wandb.log({f"TC Dimensions": [wandb.Image(plt)]})

    # ===================================
    # EXPERIMENT II - SAMPLES vs TIME
    # ===================================

    times = list()
    n_dimensions = 10
    n_samples = [100, 1_000, 10_000, 100_000, 500_000, 1_000_000]
    n_layers = 100
    tc_true, tc_est = [], []

    with tqdm.tqdm(n_samples, desc="Samples") as pbar:
        for i_samples in pbar:
            fake_data = rng.randn(i_samples, n_dimensions)

            # Generate random Data
            A = rng.rand(n_dimensions, n_dimensions)

            fake_data = fake_data @ A

            # covariance matrix
            C = A.T @ A
            vv = onp.diag(C)
            tc = onp.log(onp.sqrt(vv)).sum() - 0.5 * onp.log(onp.linalg.det(C))
            wandb.log({"sample_tc_true": tc}, step=i_samples)
            tc_true.append(tc)
            t0 = time.time()

            # define marginal entropy function
            nbins = int(onp.ceil(onp.sqrt(fake_data.shape[0])))
            entropy_f = jax.partial(histogram_entropy, nbins=nbins, base=2)

            # define marginal uniformization function
            hist_transform_f = jax.partial(histogram_transform, nbins=nbins)

            # find the total correlation
            X_trans, loss = rbig_total_correlation(
                np.array(fake_data).block_until_ready(),
                marginal_uni=hist_transform_f,
                uni_entropy=entropy_f,
                n_iterations=n_layers,
            )
            tc = onp.array(np.sum(loss) * np.log(2))

            t1 = time.time() - t0
            wandb.log({"time": t1}, step=i_samples)
            wandb.log({"sample_tc_est": tc}, step=i_samples)
            tc_est.append(tc)
            times.append(t1)

    fig, ax = plt.subplots()
    ax.plot(n_samples, times)
    ax.set(xlabel="Samples", ylabel="Timing (seconds)")
    plt.yscale("log")
    plt.tight_layout()
    wandb.log({f"Samples": [wandb.Image(plt)]})

    fig, ax = plt.subplots()
    ax.plot(n_samples, tc_est, label="Estimated")
    ax.plot(n_samples, tc_true, label="True")
    ax.set(xlabel="Samples", ylabel="Timing (seconds)")
    plt.tight_layout()
    wandb.log({f"TC Samples": [wandb.Image(plt)]})


if __name__ == "__main__":
    main()
