from typing import Callable, List, Optional, Tuple

import jax.numpy as jnp
from chex import Array, dataclass

from rbig_jax.transforms.histogram import InitUniHistTransform
from rbig_jax.transforms.inversecdf import InitInverseGaussCDF
from rbig_jax.transforms.kde import InitUniKDETransform
from rbig_jax.transforms.rotation import InitPCARotation
from flax import struct


@struct.dataclass
class RBIGBlockInit:
    init_functions: List[dataclass]

    def forward(self, inputs: Array) -> Array:
        outputs = inputs
        for i_init_f in self.init_functions:
            outputs = i_init_f.transform(outputs)
        return outputs

    def forward_gradient_bijector(self, inputs: Array) -> Tuple[Array, Array]:
        outputs = inputs
        bijectors = []
        total_logabsdet = jnp.zeros_like(outputs)

        # loop through bijectors
        for i_init_f in self.init_functions:

            # transform and params
            outputs, logabsdet, ibijector = i_init_f.transform_gradient_bijector(
                outputs
            )

            # accumulate params
            bijectors.append(ibijector)
            total_logabsdet += logabsdet

        return outputs, total_logabsdet, bijectors

    def forward_and_bijector(self, inputs: Array) -> Tuple[Array, Array]:
        outputs = inputs
        bijectors = []

        # loop through bijectors
        for i_init_f in self.init_functions:

            # transform and params
            outputs, ibijector = i_init_f.transform_and_bijector(outputs)

            # accumulate params
            bijectors.append(ibijector)

        return outputs, bijectors


@struct.dataclass
class RBIGBlockParams:
    support: Array
    quantiles: Array
    support_pdf: Array
    empirical_pdf: Array
    rotation: Array


def init_default_rbig_block(
    shape: Tuple,
    method: str = "histogram",
    support_extension: int = 10,
    alpha: float = 1e-5,
    precision: int = 100,
    nbins: Optional[int] = None,
    bw: str = "scott",
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
    else:
        raise ValueError(f"Unrecognzed Method : {method}")

    # init Inverse Gaussian CDF transform
    init_icdf_f = InitInverseGaussCDF(eps=eps, jitted=jitted)

    # init PCA transformation
    init_pca_f = InitPCARotation(jitted=jitted)

    return RBIGBlockInit(init_functions=[init_hist_f, init_icdf_f, init_pca_f])
