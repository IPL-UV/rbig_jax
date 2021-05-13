"""Code taken from:
https://github.com/jfcrenshaw/pzflow/blob/main/pzflow/bijectors/neural_splines.py
"""


from typing import Callable, Tuple

import jax.numpy as jnp
import jax.random as jr
from chex import Array, dataclass
from distrax._src.bijectors.rational_quadratic_spline import (
    RationalQuadraticSpline as distrax_rqs,
)
from distrax._src.bijectors.rational_quadratic_spline import (
    _rational_quadratic_spline_fwd,
    _rational_quadratic_spline_inv,
)
from jax.nn import softmax, softplus
from jax.random import PRNGKey

from rbig_jax.transforms.base import Bijector, InitLayersFunctions
from flax import struct


@struct.dataclass
class RationalQuadraticSpline(Bijector):
    x_pos: Array
    y_pos: Array
    knot_slopes: Array

    def forward_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:
        fn = jnp.vectorize(
            _rational_quadratic_spline_fwd, signature="(),(n),(n),(n)->(),()"
        )
        outputs, log_det = fn(inputs, self.x_pos, self.y_pos, self.knot_slopes)
        return outputs, log_det

    def inverse_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:
        fn = jnp.vectorize(
            _rational_quadratic_spline_inv, signature="(),(n),(n),(n)->(),()"
        )
        outputs, log_det = fn(inputs, self.x_pos, self.y_pos, self.knot_slopes)
        return outputs, log_det


def InitPiecewiseRationalQuadraticCDF(
    n_bins: int,
    range_min: float,
    range_max: float,
    identity_init: bool = False,
    boundary_slopes: str = "identity",
    min_bin_size: float = 1e-4,
    min_knot_slope: float = 1e-4,
):
    # preliminary checks of parameters
    if range_min >= range_max:
        raise ValueError(
            f"`range_min` is less than or equal to `range_max`; "
            f"Got: {range_min} and {range_max}"
        )
    if min_bin_size <= 0:
        raise ValueError(f"Minimum bin size must be positive; " f"Got {min_bin_size}")
    if min_knot_slope <= 0:
        raise ValueError(
            f"Minimum knot slope must be positive; " f"Got {min_knot_slope}"
        )

    def bijector(
        inputs: Array = None, rng: PRNGKey = None, shape: int = None, **kwargs
    ) -> Bijector:

        bijector = init_spline_params(
            n_bins=n_bins,
            rng=rng,
            shape=shape,
            identity_init=identity_init,
            min_knot_slope=min_knot_slope,
            range_min=range_min,
            range_max=range_max,
            boundary_slopes=boundary_slopes,
        )

        return bijector

    def transform_and_bijector(
        inputs: Array = None, rng: PRNGKey = None, shape: int = None, **kwargs
    ) -> Tuple[Array, Bijector]:

        # init bijector
        bijector = init_spline_params(
            n_bins=n_bins,
            rng=rng,
            shape=shape,
            identity_init=identity_init,
            min_knot_slope=min_knot_slope,
            range_min=range_min,
            range_max=range_max,
            boundary_slopes=boundary_slopes,
        )
        # forward transform
        outputs = bijector.forward(inputs=inputs)

        return outputs, bijector

    def transform(
        inputs: Array = None, rng: PRNGKey = None, shape: int = None, **kwargs
    ) -> Array:

        # init bijector
        bijector = init_spline_params(
            n_bins=n_bins,
            rng=rng,
            shape=shape,
            identity_init=identity_init,
            min_knot_slope=min_knot_slope,
            range_min=range_min,
            range_max=range_max,
            boundary_slopes=boundary_slopes,
        )
        outputs = bijector.forward(inputs=inputs)

        return outputs

    def transform_gradient_bijector(
        inputs: Array = None, rng: PRNGKey = None, shape: int = None, **kwargs
    ) -> Array:

        # init bijector
        bijector = init_spline_params(
            n_bins=n_bins,
            rng=rng,
            shape=shape,
            identity_init=identity_init,
            min_knot_slope=min_knot_slope,
            range_min=range_min,
            range_max=range_max,
            boundary_slopes=boundary_slopes,
        )
        outputs, logabsdet = bijector.forward_and_log_det(inputs=inputs)

        return outputs, logabsdet, bijector

    return InitLayersFunctions(
        transform=transform,
        bijector=bijector,
        transform_and_bijector=transform_and_bijector,
        transform_gradient_bijector=transform_gradient_bijector,
    )


def init_spline_params(
    n_bins: int,
    rng: PRNGKey,
    shape: Tuple[int],
    range_min: float,
    range_max: float,
    identity_init: bool = False,
    boundary_slopes: str = "identity",
    min_bin_size: float = 1e-4,
    min_knot_slope: float = 1e-4,
    dtype: type = jnp.float32,
):
    if isinstance(shape, int):
        shape = (shape,)

    if identity_init:
        # initialize widths and heights
        unnormalized_widths = jnp.zeros(shape=(*shape, n_bins), dtype=dtype)
        unnormalized_heights = jnp.zeros(shape=(*shape, n_bins), dtype=dtype)

        # initialize derivatives
        constant = jnp.log(jnp.exp(1 - min_knot_slope) - 1)
        unnormalized_derivatives = constant * jnp.ones(
            shape=(*shape, n_bins + 1), dtype=dtype
        )
    else:
        init_key = jr.split(rng, num=3)

        # initialize widths and heights
        unnormalized_widths = jr.uniform(
            key=init_key[0], minval=0, maxval=1, shape=(*shape, n_bins)
        )
        unnormalized_heights = jr.uniform(
            key=init_key[1], minval=0, maxval=1, shape=(*shape, n_bins)
        )

        # initialize derivatives
        unnormalized_derivatives = jr.uniform(
            key=init_key[2], minval=0, maxval=1, shape=(*shape, n_bins + 1)
        )
    params = jnp.concatenate(
        [unnormalized_widths, unnormalized_heights, unnormalized_derivatives], axis=-1
    )

    clf = distrax_rqs(
        params,
        range_min=range_min,
        range_max=range_max,
        boundary_slopes=boundary_slopes,
        min_bin_size=min_bin_size,
        min_knot_slope=min_knot_slope,
    )
    init_rqs_bijector = RationalQuadraticSpline(
        x_pos=clf.x_pos, y_pos=clf.y_pos, knot_slopes=clf.knot_slopes
    )
    return init_rqs_bijector
