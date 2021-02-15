import objax
from objax import TrainVar, TrainRef, StateVar
import jax.numpy as np
from objax.typing import JaxArray
from jax.nn import log_softmax, softplus, sigmoid


class Logit(objax.Module):
    def __init__(self, temperature=1, eps=1e-6, learn_temperature=False):
        super().__init__()
        self.eps = StateVar(np.array(eps))
        if learn_temperature:
            self.temperature = TrainVar(np.array(temperature))
        else:
            self.temperature = StateVar(np.array(temperature))

    def __call__(self, inputs):

        inputs = np.clip(inputs, self.eps.value, 1 - self.eps.value)

        outputs = (1 / self.temperature.value) * (np.log(inputs) - np.log1p(-inputs))
        logabsdet = -(
            np.log(self.temperature.value)
            - softplus(-self.temperature.value * outputs)
            - softplus(self.temperature.value * outputs)
        )
        return outputs, logabsdet

    def inverse(self, inputs: JaxArray) -> JaxArray:
        inputs = inputs * self.temperature.value
        outputs = sigmoid(inputs)

        # logabsdet = np.log(self.temperature) - softplus(-inputs) - softplus(inputs)
        return outputs
