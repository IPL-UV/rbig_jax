from typing import Callable, Optional, Tuple, List

import jax.numpy as np
import jax.numpy as jnp
from chex import Array, dataclass

# from rbig_jax.transforms.histogram import InitUniHistUniformize
# from rbig_jax.transforms.inversecdf import InitInverseGaussCDF
# from rbig_jax.transforms.marginal import (
#     marginal_fit_transform,
#     marginal_gradient_transform,
#     marginal_transform,
# )
from rbig_jax.transforms.rotation import InitPCARotation


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
