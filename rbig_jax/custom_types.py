"""
Taken from: 
https://github.com/aidanscannell/GPJax/blob/master/gpjax/custom_types.py
"""
from typing import Tuple, Union

from jax import numpy as jnp

MeanAndVariance = Tuple[jnp.ndarray, jnp.ndarray]
InputData = jnp.ndarray
OutputData = jnp.ndarray
MeanFunc = jnp.float64

Variance = Union[jnp.float64, jnp.ndarray]
Lengthscales = Union[jnp.float64, jnp.ndarray]
