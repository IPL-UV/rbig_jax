import jax.numpy as jnp
from chex import dataclass, Array
from typing import NamedTuple, Tuple, Callable


def cascade_forward(bijectors: dataclass, inputs: Array) -> Tuple[Array, Array]:
    outputs = inputs
    for ibijector in bijectors:
        outputs = ibijector.forward(outputs)
    return outputs


def cascade_forward_and_log_det(
    bijectors: dataclass, inputs: Array
) -> Tuple[Array, Array]:
    outputs = inputs
    total_logabsdet = jnp.zeros_like(outputs)
    for ibijector in bijectors:
        outputs, logabsdet = ibijector.forward_and_log_det(outputs)
        total_logabsdet += logabsdet
    return outputs, total_logabsdet


def cascade_inverse_and_log_det(
    bijectors: dataclass, inputs: Array
) -> Tuple[Array, Array]:
    outputs = inputs
    total_logabsdet = jnp.zeros_like(outputs)
    for ibijector in reversed(bijectors):
        outputs, logabsdet = ibijector.inverse_and_log_det(outputs)
        total_logabsdet += logabsdet
    return outputs, total_logabsdet


def cascade_inverse(bijectors: dataclass, inputs: Array) -> Tuple[Array, Array]:
    outputs = inputs
    for ibijector in reversed(bijectors):
        outputs = ibijector.inverse(outputs)
    return outputs


class _CascadeTransform(NamedTuple):
    forward: Callable
    inverse: Callable
    forward_and_log_det: Callable
    inverse_and_log_det: Callable


CascadeTransform = _CascadeTransform(
    forward=cascade_forward,
    inverse=cascade_inverse,
    forward_and_log_det=cascade_forward_and_log_det,
    inverse_and_log_det=cascade_inverse_and_log_det,
)

