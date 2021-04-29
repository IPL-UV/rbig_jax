from typing import Optional

import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid

from rbig_jax.transforms.reshape import unflatten_image

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)


def plot_joint(
    data: np.ndarray,
    color: str = "red",
    title: str = "",
    kind="kde",
    logger=None,
    save_name=None,
):

    plt.figure()
    sns.jointplot(x=data[:, 0], y=data[:, 1], kind=kind, color=color)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.suptitle(title)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name)
    else:
        plt.show()


def plot_joint_prob(
    data: np.ndarray,
    probs: np.ndarray,
    cmap="Reds",
    title="",
    logger=None,
    save_name=None,
):

    fig, ax = plt.subplots()
    h = ax.scatter(data[:, 0], data[:, 1], s=1, c=probs, cmap=cmap)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(h,)
    ax.set_title(title)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name)
    else:
        plt.show()


def plot_info_loss(
    data: np.ndarray,
    n_layers: str = None,
    title: str = "Information Loss",
    logger=None,
    save_name=None,
):

    # plt.figure()
    # plt.plot(data)
    # plt.xlabel("Layers")
    # plt.ylabel("Change in Total Correlation")
    # plt.title(title)
    # plt.tight_layout()
    # plt.show()
    title = (
        f"Information Loss, n={n_layers}"
        if n_layers is not None
        else "Information Loss"
    )
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set(xlabel="Layers", ylabel="Sum of Marginal Entropy", title=title)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name)
    else:
        plt.show()




def plot_image_grid(image, image_shape: Optional = None):

    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(5, 5),  # creates 2x2 grid of axes
        axes_pad=0.1,  # pad between axes in inch.
    )

    for i, ax in enumerate(grid):
        img = image[i]

        if image_shape is not None:
            img = unflatten_image(img, image_shape, batch=False)

        ax.imshow(img, cmap="gray")

    plt.show
    return fig, grid
