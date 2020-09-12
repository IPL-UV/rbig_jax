import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)


def plot_joint(
    data: np.ndarray, color: str = "red", title: str = "", kind="kde", logger=None
):

    plt.figure()
    sns.jointplot(x=data[:, 0], y=data[:, 1], kind=kind, color=color)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_joint_prob(
    data: np.ndarray, probs: np.ndarray, cmap="Reds", title="", logger=None
):

    fig, ax = plt.subplots()
    h = ax.scatter(data[:, 0], data[:, 1], s=1, c=probs, cmap=cmap)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(h,)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_info_loss(
    data: np.ndarray, n_layers: str = None, title: str = "Information Loss", logger=None
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
    plt.show()
