from typing import Callable, List, Sequence, Tuple

import jax
import jax.numpy as np
import objax
from objax.module import Module
from objax.typing import JaxArray
from jax.random import PRNGKey


class Transform(Module):
    """Base class for all transformation"""

    def forward(self, inputs: JaxArray) -> Tuple[JaxArray, JaxArray]:
        raise NotImplementedError()

    def inverse(self, inputs: JaxArray) -> JaxArray:
        raise NotImplementedError


def CompositeTransform(bijectors: Sequence[Callable]):
    def init_fun(rng: PRNGKey, n_features: int, **kwargs):

        # initialize params stores
        all_params, forward_funs, inverse_funs = [], [], []
        # create keys
        rng, *layer_rngs = jax.random.split(rng, num=len(bijectors) + 1)
        for i_rng, init_f in zip(layer_rngs, bijectors):

            param, forward_f, inverse_f = init_f(rng=i_rng, n_features=n_features)

            all_params.append(param)
            forward_funs.append(forward_f)
            inverse_funs.append(inverse_f)

        def bijector_chain(params, bijectors, inputs, **kwargs):
            logabsdet = np.zeros(inputs.shape[0])
            for bijector, param in zip(bijectors, params):
                inputs, ilogabsdet = bijector(param, inputs, **kwargs)
                logabsdet += ilogabsdet
            return inputs, logabsdet

        def forward_func(params, inputs, **kwargs):
            return bijector_chain(params, forward_funs, inputs, **kwargs)

        def inverse_func(params, inputs, **kwargs):
            return bijector_chain(params[::-1], inverse_funs[::-1], inputs, **kwargs)

        return all_params, forward_func, inverse_func

    return init_fun


# class CompositeTransform(Transform):
#     def __init__(self, transforms: List[Module]) -> None:
#         super().__init__()
#         self._transforms = objax.ModuleList(transforms)

#     def __call__(self, inputs: JaxArray) -> Tuple[JaxArray, JaxArray]:
#         outputs = inputs
#         total_logabsdet = np.zeros_like(outputs.shape[0])
#         for transform in self._transforms:
#             outputs, logabsdet = transform(outputs)
#             total_logabsdet += logabsdet
#         return outputs, total_logabsdet

#     def transform(self, inputs: JaxArray) -> Tuple[JaxArray, JaxArray]:
#         outputs = inputs
#         for itransform in self._transforms:
#             outputs = itransform.transform(outputs)
#         return outputs

#     def inverse(self, inputs: JaxArray) -> Tuple[JaxArray, JaxArray]:
#         outputs = inputs
#         for itransform in self._transforms[::-1]:
#             outputs = itransform.inverse(outputs)
#         return outputs
