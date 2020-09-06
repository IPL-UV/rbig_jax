import matplotlib.pyplot as plt
import seaborn as sns

import wandb

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)


def plot_joint(data, color: str = "red", title: str = "", kind="kde", logger=None):

    plt.figure()
    sns.jointplot(x=data[:, 0], y=data[:, 1], kind=kind, color=color)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.suptitle(title)
    plt.tight_layout()
    if logger is not None:
        wandb.log({title: [wandb.Image(plt)]})
        plt.gcf()
        plt.clf()
        plt.close()
    else:
        plt.show()
