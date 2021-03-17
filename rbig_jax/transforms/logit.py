from typing import Tuple, Callable, Optional

import jax.numpy as np
from jax.random import PRNGKey
from jax.nn import sigmoid, softplus
from chex import dataclass, Array


def Logit(eps: float = 1e-5, temperature: float = 1.0):
    def init_func(rng: PRNGKey, n_features: int, **kwargs):
        def forward_func(
            params: Optional[dataclass], inputs: Array, **kwargs
        ) -> Tuple[Array, Array]:

            inputs = np.clip(inputs, eps, 1 - eps)

            outputs = (1 / temperature) * (np.log(inputs) - np.log1p(-inputs))
            logabsdet = -(
                np.log(temperature)
                - softplus(-temperature * outputs)
                - softplus(temperature * outputs)
            )

            return outputs, logabsdet.sum(axis=1)

        def inverse_func(
            params: Optional[dataclass], inputs: Array, **kwargs
        ) -> Tuple[Array, Array]:

            inputs = inputs * temperature
            outputs = sigmoid(inputs)

            logabsdet = -(
                np.log(temperature)
                - softplus(-temperature * outputs)
                - softplus(temperature * outputs)
            )

            return outputs, logabsdet.sum(axis=1)

        return (), forward_func, inverse_func

    return init_func
