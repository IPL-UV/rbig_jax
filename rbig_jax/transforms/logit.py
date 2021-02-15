import objax
from objax import TrainVar, TrainRef
import jax.numpy as np
from objax.typing import JaxArray
from jax.nn import log_softmax, softplus, sigmoid


class Logit(objax.Module):
    def __init__(self, temperature=1, eps=1e-6, learn_temperature=False):
        super().__init__()
        self.eps = eps  # TrainRef(np.asarray(eps))
        # if learn_temperature:
        #     self.temperature = nn.Parameter(torch.Tensor([temperature]))
        # else:
        self.temperature = TrainVar(np.array(temperature))

    # def forward(self, inputs, context=None):
    #     inputs = self.temperature * inputs
    #     outputs = torch.sigmoid(inputs)
    #     logabsdet = torchutils.sum_except_batch(
    #         torch.log(self.temperature) - F.softplus(-inputs) - F.softplus(inputs)
    #     )
    #     return outputs, logabsdet

    def __call__(self, inputs):

        inputs = np.clip(inputs, self.eps, 1 - self.eps)

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
