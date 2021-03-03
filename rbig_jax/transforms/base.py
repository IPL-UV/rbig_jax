from typing import List, Tuple

import jax
import jax.numpy as np
import objax
from objax.module import Module
from objax.typing import JaxArray


class Transform(Module):
    """Base class for all transformation"""

    def forward(self, inputs: JaxArray) -> Tuple[JaxArray, JaxArray]:
        raise NotImplementedError()

    def inverse(self, inputs: JaxArray) -> JaxArray:
        raise NotImplementedError


class CompositeTransform(Transform):
    def __init__(self, transforms: List[Module]) -> None:
        super().__init__()
        self._transforms = objax.nn.Sequential(transforms)

    def __call__(self, inputs: JaxArray) -> Tuple[JaxArray, JaxArray]:
        outputs = inputs
        total_logabsdet = np.zeros_like(outputs.shape[0])
        for transform in self._transforms:
            outputs, logabsdet = transform(outputs)
            total_logabsdet += logabsdet
        return outputs, total_logabsdet

    def transform(self, inputs: JaxArray) -> Tuple[JaxArray, JaxArray]:
        outputs = inputs
        for itransform in self._transforms:
            outputs = itransform.transform(outputs)
        return outputs

    def inverse(self, inputs: JaxArray) -> Tuple[JaxArray, JaxArray]:
        outputs = inputs
        for itransform in self._transforms[::-1]:
            outputs = itransform.inverse(outputs)
        return outputs
