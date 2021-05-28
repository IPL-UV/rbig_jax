from typing import Iterable, Optional, Tuple

import jax.numpy as jnp
from chex import Array, dataclass
from distrax._src.distributions.distribution import Distribution
from flax import struct

from rbig_jax.models.gaussflow import GaussianizationFlow
from rbig_jax.transforms.base import Bijector
from rbig_jax.transforms.block import RBIGBlockInit
from rbig_jax.transforms.histogram import InitUniHistTransform
from rbig_jax.transforms.inversecdf import InitInverseGaussCDF
from rbig_jax.transforms.kde import InitUniKDETransform
from rbig_jax.transforms.parametric.mixture.gaussian import InitMixtureGaussianCDF
from rbig_jax.transforms.rotation import InitPCARotation


@struct.dataclass
class RBIGFlow(GaussianizationFlow):
    bijectors: Iterable[Bijector]
    base_dist: Distribution = struct.field(pytree_node=False)
    info_loss: Array = struct.field(pytree_node=False)


def init_default_rbig_block(
    shape: Tuple,
    method: str = "histogram",
    support_extension: int = 10,
    alpha: float = 1e-5,
    precision: int = 100,
    nbins: Optional[int] = None,
    bw: str = "scott",
    n_components: int = 15,
    jitted: bool = True,
    eps: float = 1e-5,
) -> RBIGBlockInit:

    n_samples = shape[0]

    # init histogram transformation
    if method == "histogram":
        if nbins is None:
            nbins = int(jnp.sqrt(n_samples))

        init_hist_f = InitUniHistTransform(
            n_samples=n_samples,
            nbins=nbins,
            support_extension=support_extension,
            precision=precision,
            alpha=alpha,
            jitted=jitted,
        )
    elif method == "kde":
        init_hist_f = InitUniKDETransform(
            shape=shape,
            support_extension=support_extension,
            precision=precision,
            bw=bw,
            jitted=jitted,
        )
    elif method == "gmm":
        init_hist_f = InitMixtureGaussianCDF(
            n_components=n_components, init_method="gmm",
        )
    else:
        raise ValueError(f"Unrecognzed Method : {method}")

    # init Inverse Gaussian CDF transform
    init_icdf_f = InitInverseGaussCDF(eps=eps, jitted=jitted)

    # init PCA transformation
    init_pca_f = InitPCARotation(jitted=jitted)

    return [init_hist_f, init_icdf_f, init_pca_f]


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
    zero_tolerance: int = 10,
    p: float = 0.25,
    verbose: bool = True,
    n_layers_remove: int = 0,
    interval: int = 5,
):

    # init rbig_block
    rbig_block_init_fs = init_default_rbig_block(
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

    rbig_block_init = RBIGBlockInit(init_functions=rbig_block_init_fs)

    from rbig_jax.losses import init_info_loss

    # initialize info loss function
    loss = init_info_loss(
        n_samples=X.shape[0],
        max_layers=max_layers,
        zero_tolerance=zero_tolerance,
        p=p,
        jitted=jitted,
    )

    from rbig_jax.training.iterative import train_info_loss_model

    # run iterative training
    X_g, rbig_model = train_info_loss_model(
        X=X,
        rbig_block_init=rbig_block_init,
        loss=loss,
        verbose=verbose,
        interval=interval,
        n_layers_remove=n_layers_remove,
        zero_tolerance=zero_tolerance,
    )
    return X_g, rbig_model
