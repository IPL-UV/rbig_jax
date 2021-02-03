# JAX SETTINGS
import time as time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as np

# Plot Functions
import matplotlib.pyplot as plt
import numpy as onp
import seaborn as sns
import tqdm
from jax.config import config
from scipy.stats import beta

from rbig_jax.data import get_classic
from rbig_jax.plots import plot_info_loss, plot_joint
from rbig_jax.information.entropy import histogram_entropy
from rbig_jax.transforms.histogram import histogram_transform
from rbig_jax.information.total_corr import total_corr_f

import wandb

config.update("jax_enable_x64", True)

# plot methods
import matplotlib.pyplot as plt
import seaborn as sns

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
    # EXPERIMENT I - DIMS vs TIME
    # ===================================

    times = list()
    dimensions = [2, 5, 10, 25, 50, 100, 1_000]
    n_layers = 100
    n_samples = 10_000

    with tqdm.tqdm(dimensions, desc="Dimensions") as pbar:
        for idim in pbar:

            fake_data = rng.randn(n_samples, idim)

            # Generate random Data
            A = rng.rand(idim, idim)

            fake_data = fake_data @ A
            # covariance matrix
            C = A.T @ A
            vv = onp.diag(C)
            tc_true = onp.log(onp.sqrt(vv)).sum() - 0.5 * onp.log(onp.linalg.det(C))
            wandb.log({"tc_true": tc_true}, step=idim)

            t0 = time.time()

            # define marginal entropy function
            nbins = int(onp.ceil(onp.sqrt(fake_data.shape[0])))
            entropy_f = jax.partial(histogram_entropy, nbins=nbins, base=2)

            # define marginal uniformization function
            hist_transform_f = jax.partial(histogram_transform, nbins=nbins)

            # find the total correlation
            X_trans, loss = total_corr_f(
                np.array(fake_data).block_until_ready(),
                marginal_uni=hist_transform_f,
                marginal_entropy=entropy_f,
                n_iterations=n_layers,
            )

            tc = onp.array(np.sum(loss) * np.log(2))

            t1 = time.time() - t0
            wandb.log({"time": t1}, step=idim)
            wandb.log({"tc_est": tc}, step=idim)
            times.append(t1)

    fig, ax = plt.subplots()
    ax.plot(dimensions, times)
    ax.set(xlabel="Dimensions", ylabel="Timing (seconds)")
    plt.yscale("log")
    plt.tight_layout()
    wandb.log({f"Dimensions": [wandb.Image(plt)]})

    # ===================================
    # EXPERIMENT I - SAMPLES vs TIME
    # ===================================

    times = list()
    n_dimensions = 10
    n_samples = [100, 1_000, 10_000, 100_000, 500_000, 1_000_000]
    n_layers = 100

    with tqdm.tqdm(n_samples, desc="Samples") as pbar:
        for i_samples in pbar:
            fake_data = rng.randn(i_samples, n_dimensions)

            # Generate random Data
            A = rng.rand(n_dimensions, n_dimensions)

            fake_data = fake_data @ A

            # covariance matrix
            C = A.T @ A
            vv = onp.diag(C)
            tc_true = onp.log(onp.sqrt(vv)).sum() - 0.5 * onp.log(onp.linalg.det(C))
            wandb.log({"tc_true": tc_true}, step=i_samples)
            t0 = time.time()

            # define marginal entropy function
            nbins = int(onp.ceil(onp.sqrt(fake_data.shape[0])))
            entropy_f = jax.partial(histogram_entropy, nbins=nbins, base=2)

            # define marginal uniformization function
            hist_transform_f = jax.partial(histogram_transform, nbins=nbins)

            # find the total correlation
            X_trans, loss = total_corr_f(
                np.array(fake_data).block_until_ready(),
                marginal_uni=hist_transform_f,
                marginal_entropy=entropy_f,
                n_iterations=n_layers,
            )
            tc = onp.array(np.sum(loss) * np.log(2))

            t1 = time.time() - t0
            wandb.log({"time": t1}, step=i_samples)
            wandb.log({"tc_est": tc}, step=i_samples)
            times.append(t1)

    fig, ax = plt.subplots()
    ax.plot(n_samples, times)
    ax.set(xlabel="Dimensions", ylabel="Timing (seconds)")
    plt.yscale("log")
    plt.tight_layout()
    wandb.log({f"Samples": [wandb.Image(plt)]})


if __name__ == "__main__":
    main()
