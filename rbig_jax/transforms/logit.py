import jax.numpy as np
import objax
from jax.nn import log_softmax, sigmoid, softplus
from objax import StateVar, TrainRef, TrainVar
from objax.typing import JaxArray

from rbig_jax.transforms.base import Transform


class Logit(Transform):
    def __init__(self, temperature=1, eps=1e-6, learn_temperature=False):
        super().__init__()
        self.eps = StateVar(np.array(eps))
        if learn_temperature:
            self.temperature = TrainVar(np.array(float(temperature)))
        else:
            self.temperature = StateVar(np.array(float(temperature)))

    def __call__(self, inputs):

        inputs = np.clip(inputs, self.eps.value, 1 - self.eps.value)

        outputs = (1 / self.temperature.value) * (np.log(inputs) - np.log1p(-inputs))
        logabsdet = -(
            np.log(self.temperature.value)
            - softplus(-self.temperature.value * outputs)
            - softplus(self.temperature.value * outputs)
        )
        return outputs, logabsdet.sum(axis=1)

    def transform(self, inputs):

        inputs = np.clip(inputs, self.eps.value, 1 - self.eps.value)

        outputs = (1 / self.temperature.value) * (np.log(inputs) - np.log1p(-inputs))
        return outputs

    def inverse(self, inputs: JaxArray) -> JaxArray:
        inputs = inputs * self.temperature.value
        outputs = sigmoid(inputs)

        # logabsdet = np.log(self.temperature) - softplus(-inputs) - softplus(inputs)
        return outputs
