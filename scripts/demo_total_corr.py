# jax packages
from functools import partial
from pathlib import Path

import chex
import jax
import jax.numpy as np
# plot methods
import matplotlib.pyplot as plt
import numpy as onp
import seaborn as sns
# logging
import tqdm
from jax.config import config

import wandb
# library functions
from rbig_jax.data import get_classic
from rbig_jax.information.entropy import histogram_entropy
from rbig_jax.information.total_corr import total_corr_f
from rbig_jax.plots import plot_info_loss, plot_joint
from rbig_jax.transforms.histogram import histogram_transform

config.update("jax_enable_x64", True)





sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)

FIG_PATH = "figures/"


def main():

    data = get_classic(100_000)

    # plot data
    plot_joint(
        data[:1_000],
        "blue",
        "Original Data",
        kind="kde",
        save_name=str(Path(FIG_PATH).joinpath("joint_data.png")),
    )

    # define marginal entropy function
    entropy_f = jax.partial(histogram_entropy, nbins=1_000, base=2)

    # define marginal uniformization function
    hist_transform_f = jax.partial(histogram_transform, nbins=1_000)

    n_iterations = 100

    X_trans, loss = total_corr_f(
        np.array(data).block_until_ready(),
        marginal_uni=hist_transform_f,
        marginal_entropy=entropy_f,
        n_iterations=n_iterations,
    )

    total_corr = np.sum(loss) * np.log(2)

    plot_info_loss(
        loss,
        n_layers=len(loss),
        save_name=str(Path(FIG_PATH).joinpath("info_loss.png")),
    )

    print(f"Total Correlation: {total_corr}")

    X_plot = onp.array(X_trans)

    plot_joint(
        X_plot[:10_000],
        "blue",
        "Latent Space",
        kind="kde",
        save_name=str(Path(FIG_PATH).joinpath("joint_latent.png")),
    )

    pass


if __name__ == "__main__":
    main()
