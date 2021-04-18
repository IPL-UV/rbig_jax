"""Code taken from:
https://github.com/jfcrenshaw/pzflow/blob/main/pzflow/bijectors/neural_splines.py
"""

from rbig_jax.transforms.base import Bijector
from typing import Tuple, Callable
from jax.random import PRNGKey
import jax.random as jr
import jax.numpy as jnp
from jax.nn import softmax, softplus
from chex import dataclass, Array
from distrax._src.bijectors.rational_quadratic_spline import (
    RationalQuadraticSpline as distrax_rqs,
    _rational_quadratic_spline_fwd,
    _rational_quadratic_spline_inv,
)


@dataclass
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


def PiecewiseRationalQuadraticCDF(
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

    def init_func(rng: PRNGKey, n_features: int, **kwargs) -> Bijector:

        return init_spline_params(
            n_bins=n_bins,
            rng=rng,
            n_features=n_features,
            identity_init=identity_init,
            min_knot_slope=min_knot_slope,
            range_min=range_min,
            range_max=range_max,
            boundary_slopes=boundary_slopes,
        )

    return init_func


def init_spline_params(
    n_bins: int,
    rng: PRNGKey,
    n_features: Tuple[int],
    range_min: float,
    range_max: float,
    identity_init: bool = False,
    boundary_slopes: str = "identity",
    min_bin_size: float = 1e-4,
    min_knot_slope: float = 1e-4,
    dtype: type = jnp.float32,
):
    if isinstance(n_features, int):
        n_features = (n_features,)

    if identity_init:
        # initialize widths and heights
        unnormalized_widths = jnp.zeros(shape=(*n_features, n_bins), dtype=dtype)
        unnormalized_heights = jnp.zeros(shape=(*n_features, n_bins), dtype=dtype)

        # initialize derivatives
        constant = jnp.log(jnp.exp(1 - min_knot_slope) - 1)
        unnormalized_derivatives = constant * jnp.ones(
            shape=(*n_features, n_bins + 1), dtype=dtype
        )
    else:
        init_key = jr.split(rng, num=3)

        # initialize widths and heights
        unnormalized_widths = jr.uniform(
            key=init_key[0], minval=0, maxval=1, shape=(*n_features, n_bins)
        )
        unnormalized_heights = jr.uniform(
            key=init_key[1], minval=0, maxval=1, shape=(*n_features, n_bins)
        )

        # initialize derivatives
        unnormalized_derivatives = jr.uniform(
            key=init_key[2], minval=0, maxval=1, shape=(*n_features, n_bins + 1)
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

