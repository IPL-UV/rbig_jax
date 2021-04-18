"""Code taken from:
https://github.com/jfcrenshaw/pzflow/blob/main/pzflow/bijectors/neural_splines.py
"""

from typing import Tuple, Callable
from jax.random import PRNGKey
import jax.random as jr
import jax.numpy as jnp
from jax.nn import softmax, softplus
from chex import dataclass, Array
from distrax._src.bijectors.rational_quadratic_spline import (
    RationalQuadraticSpline,
    _rational_quadratic_spline_fwd,
    _rational_quadratic_spline_inv,
    _normalize_bin_sizes,
    _normalize_knot_slopes,
)


@dataclass
class RQSplineParams:
    x_pos: Array
    y_pos: Array
    knot_slopes: Array


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

    def init_func(
        rng: PRNGKey, n_features: int, **kwargs
    ) -> Tuple[RQSplineParams, Callable, Callable]:

        init_params = init_spline_params(
            n_bins=n_bins,
            rng=rng,
            n_features=n_features,
            identity_init=identity_init,
            min_knot_slope=min_knot_slope,
            range_min=range_min,
            range_max=range_max,
            boundary_slopes=boundary_slopes,
        )

        def forward_func(
            params: RQSplineParams, inputs: Array, **kwargs
        ) -> Tuple[Array, Array]:
            fn = jnp.vectorize(
                _rational_quadratic_spline_fwd, signature="(),(n),(n),(n)->(),()"
            )
            outputs, log_det = fn(
                inputs, params.x_pos, params.y_pos, params.knot_slopes
            )
            return outputs, log_det

        def inverse_func(
            params: RQSplineParams, inputs: Array, **kwargs
        ) -> Tuple[Array, Array]:
            fn = jnp.vectorize(
                _rational_quadratic_spline_inv, signature="(),(n),(n),(n)->(),()"
            )
            outputs, log_det = fn(
                inputs, params.x_pos, params.y_pos, params.knot_slopes
            )
            return outputs, log_det

        return init_params, forward_func, inverse_func

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

    clf = RationalQuadraticSpline(
        params,
        range_min=range_min,
        range_max=range_max,
        boundary_slopes=boundary_slopes,
        min_bin_size=min_bin_size,
        min_knot_slope=min_knot_slope,
    )

    # # Normalize bin sizes and compute bin positions on the x and y axis.
    # range_size = range_max - range_min
    # bin_widths = _normalize_bin_sizes(unnormalized_widths, range_size, min_bin_size)
    # bin_heights = _normalize_bin_sizes(unnormalized_heights, range_size, min_bin_size)
    # x_pos = range_min + jnp.cumsum(bin_widths[..., :-1], axis=-1)
    # y_pos = range_min + jnp.cumsum(bin_heights[..., :-1], axis=-1)
    # pad_shape = n_features + (1,)
    # pad_below = jnp.full(pad_shape, range_min, dtype=dtype)
    # pad_above = jnp.full(pad_shape, range_max, dtype=dtype)
    # x_pos = jnp.concatenate([pad_below, x_pos, pad_above], axis=-1)
    # y_pos = jnp.concatenate([pad_below, y_pos, pad_above], axis=-1)
    # # Normalize knot slopes and enforce requested boundary conditions.
    # knot_slopes = _normalize_knot_slopes(unnormalized_derivatives, min_knot_slope)
    # if boundary_slopes == "unconstrained":
    #     pass
    # elif boundary_slopes == "lower_identity":
    #     ones = jnp.ones(pad_shape, dtype)
    #     knot_slopes = jnp.concatenate([ones, knot_slopes[..., 1:]], axis=-1)
    # elif boundary_slopes == "upper_identity":
    #     ones = jnp.ones(pad_shape, dtype)
    #     knot_slopes = jnp.concatenate([knot_slopes[..., :-1], ones], axis=-1)
    # elif boundary_slopes == "identity":
    #     ones = jnp.ones(pad_shape, dtype)
    #     knot_slopes = jnp.concatenate([ones, knot_slopes[..., 1:-1], ones], axis=-1)
    # elif boundary_slopes == "circular":
    #     knot_slopes = jnp.concatenate(
    #         [knot_slopes[..., :-1], knot_slopes[..., :1]], axis=-1
    #     )
    # else:
    #     raise ValueError(
    #         f"Unknown option for boundary slopes:" f" `{boundary_slopes}`."
    #     )

    init_params = RQSplineParams(
        x_pos=clf.x_pos, y_pos=clf.y_pos, knot_slopes=clf.knot_slopes
    )
    return init_params
