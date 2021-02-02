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


config.update("jax_enable_x64", True)

# plot methods
import matplotlib.pyplot as plt
import seaborn as sns

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)

FIG_PATH = "figures/"


def main():

    times = list()
    dimensions = [2, 5, 10, 25, 50, 100, 1_000]
    n_layers = 100
    n_samples = 10_000
    total_corr = list()

    with tqdm.tqdm(dimensions, desc="Dimensions") as pbar:
        for idim in pbar:
            fake_data = onp.random.randn(n_samples, idim)
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

            total_corr.append(onp.sum(loss) * np.log(2))

            times.append(time.time() - t0)

    fig, ax = plt.subplots()
    ax.plot(dimensions, times)
    ax.set(xlabel="Dimensions", ylabel="Timing (seconds)")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(str(Path(FIG_PATH).joinpath("dim_v_time.png")))

    fig, ax = plt.subplots()
    ax.plot(dimensions, total_corr)
    ax.set(xlabel="Dimensions", ylabel="Timing (seconds)")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(str(Path(FIG_PATH).joinpath("dim_v_info.png")))

    # EXPERIMENT WITH SAMPLES

    times = list()
    n_dimensions = 10
    n_samples = [100, 1_000, 10_000, 100_000, 500_000, 1_000_000]
    n_layers = 100
    total_corr = list()

    with tqdm.tqdm(n_samples, desc="Samples") as pbar:
        for i_samples in pbar:
            fake_data = onp.random.randn(i_samples, n_dimensions)

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

            total_corr.append(onp.sum(loss) * np.log(2))
            times.append(time.time() - t0)

    fig, ax = plt.subplots()
    ax.plot(n_samples, times)
    ax.set(xlabel="Dimensions", ylabel="Timing (seconds)")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(str(Path(FIG_PATH).joinpath("samp_v_time.png")))

    fig, ax = plt.subplots()
    ax.plot(n_samples, total_corr)
    ax.set(xlabel="Dimensions", ylabel="Timing (seconds)")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(str(Path(FIG_PATH).joinpath("samp_v_info.png")))


if __name__ == "__main__":
    main()