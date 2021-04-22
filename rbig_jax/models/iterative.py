from chex import Array, dataclass
from rbig_jax.models.gaussflow import GaussianizationFlow

from typing import Optional, Iterable

from rbig_jax.transforms.base import Bijector

from distrax._src.distributions.distribution import Distribution

from rbig_jax.transforms.block import init_default_rbig_block

from rbig_jax.training.iterative import train_info_loss_model
from rbig_jax.losses import init_info_loss


# @dataclass
# class IterativeGaussianization(GaussianizationFlow):
#     bijectors: Iterable[Bijector]
#     base_dist: Distribution
#     info_loss: Array
#     pass


def RBIG(
    X: Array,
    support_extension: int = 10,
    method: str = "histogram",
    precision: int = 100,
    alpha: int = 1e-5,
    nbins: Optional[int] = None,
    bw: str = "scott",
    jitted: bool = False,
    eps: float = 1e-5,
    max_layers: int = 1_000,
    zero_tolerance: int = 30,
    p: float = 0.25,
    base: int = 2,
    verbose: bool = True,
    n_layers_remove: int = 40,
    interval: int = 5,
):

    # init rbig_block
    rbig_block = init_default_rbig_block(
        shape=X.shape,
        support_extension=support_extension,
        alpha=alpha,
        precision=precision,
        nbins=nbins,
        bw=bw,
        jitted=jitted,
        eps=eps,
        method=method,
    )

    # initialize info loss function
    loss = init_info_loss(
        n_samples=X.shape[0],
        max_layers=max_layers,
        zero_tolerance=zero_tolerance,
        p=p,
        base=base,
        jitted=jitted,
    )

    # run iterative training
    X_g, rbig_model = train_info_loss_model(
        X=X,
        rbig_block=rbig_block,
        loss=loss,
        verbose=verbose,
        interval=interval,
        n_layers_remove=n_layers_remove,
    )
    return X_g, rbig_model

