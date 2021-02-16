from typing import Callable, Tuple, Union

from objax.typing import JaxArray


def cascade_gradient(
    inputs: JaxArray, funcs: Callable
) -> Union[JaxArray, Tuple[JaxArray, JaxArray]]:
    outputs = inputs
    total_logabsdet = None
    for ifunc in funcs:
        outputs, logabsdet = ifunc(outputs)
        total_logabsdet += logabsdet
    return outputs, total_logabsdet


def cascade_transform(
    inputs: JaxArray, funcs: Callable
) -> Union[JaxArray, Tuple[JaxArray, JaxArray]]:
    outputs = inputs
    for ifunc in funcs:
        outputs = ifunc(outputs)
    return outputs
