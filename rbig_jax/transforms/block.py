from typing import Callable, Optional, Tuple, List

import jax.numpy as np
import jax.numpy as jnp
from chex import Array, dataclass
from rbig_jax.transforms.histogram import InitUniHistTransform
from rbig_jax.transforms.inversecdf import InitInverseGaussCDFTransform
from rbig_jax.transforms.rotation import InitPCARotation

# from rbig_jax.transforms.histogram import InitUniHistUniformize
# from rbig_jax.transforms.inversecdf import InitInverseGaussCDF
# from rbig_jax.transforms.marginal import (
#     marginal_fit_transform,
#     marginal_gradient_transform,
#     marginal_transform,
# )


@dataclass
class RBIGBlock:
    init_functions: List[dataclass]

    def forward_and_params(self, inputs: Array) -> Tuple[Array, Array]:
        outputs = inputs
        params = []

        # loop through bijectors
        for ibijector in self.init_functions:

            # transform and params
            outputs, iparams = ibijector.init_bijector(outputs)

            # accumulate params
            params.append(iparams)

        return outputs, params

    # def forward_and_params_and_gradient(self, inputs: Array) -> Tuple[Array, Array]:
    #     outputs = inputs
    #     params = []

    #     # loop through bijectors
    #     for ibijector in self.init_functions:

    #         # transform and params
    #         outputs, iparams = ibijector.init_bijector(outputs)

    #         # outputs

    #         # accumulate params
    #         params.append(iparams)

    #     return outputs, params

    def forward(self, inputs: Array) -> Tuple[Array, Array]:
        outputs = inputs
        for ibijector in self.init_functions:
            outputs = ibijector.init_transform(outputs)
        return outputs


@dataclass
class RBIGBlockParams:
    support: Array
    quantiles: Array
    support_pdf: Array
    empirical_pdf: Array
    rotation: Array


def init_default_rbig_block(
    n_samples: int,
    support_extension: int = 10,
    alpha: float = 1e-5,
    precision: int = 100,
    nbins: Optional[int] = None,
    jitted: bool = True,
    eps: float = 1e-5,
) -> RBIGBlock:

    # init histogram transformation
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

    # init Inverse Gaussian CDF transform
    init_icdf_f = InitInverseGaussCDFTransform(eps=eps)

    # init PCA transformation
    init_pca_f = InitPCARotation(jitted=jitted)

    return RBIGBlock(init_functions=[init_hist_f, init_icdf_f, init_pca_f])
