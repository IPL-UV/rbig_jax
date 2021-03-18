import chex
import jax
import jax.numpy as np
import numpy as onp
import objax
import pytest


from rbig_jax.transforms.parametric.splines import rational_quadratic_spline

seed = 123
rng = onp.random.RandomState(123)
generator = objax.random.Generator(123)

KEY = jax.random.PRNGKey(seed)


# @pytest.mark.parametrize("n_samples", [1, 10, 100])
# @pytest.mark.parametrize("n_bins", [1, 10, 100])
# @pytest.mark.parametrize("n_features", [1, 10, 100])
# @pytest.mark.parametrize("bounds", [1, 10, 100])
# def test_splines_shape(n_samples, n_features, n_bins, bounds):

#     # initialize function

#     data_rng, *spline_rng = jax.random.split(KEY, 2)

#     x = jax.random.normal(data_rng, shape=(n_samples, n_features))

#     # initialize the parameters
#     widths = jax.random.normal(spline_rng[0], shape=(1, n_features, n_bins))
#     heights = jax.random.normal(spline_rng[0], shape=(1, n_features, n_bins))
#     derivatives = jax.random.normal(spline_rng[0], shape=(1, n_features, n_bins))

#     # create layer
#     z, logabsdet = rational_quadratic_spline(
#         x, widths, heights, derivatives, bounds, inverse=False
#     )

#     # checks
#     chex.assert_equal_shape([z, x])
#     chex.assert_shape(logabsdet, (n_samples,))

#     # inverse transformation
#     x_approx, logabsdet = rational_quadratic_spline(
#         z, widths, heights, derivatives, bounds, inverse=True
#     )

#     # checks
#     chex.assert_equal_shape([x_approx, x])
#     chex.assert_shape(logabsdet, (n_samples,))


@pytest.mark.parametrize("n_samples", [1, 10, 100])
@pytest.mark.parametrize("n_bins", [1, 10, 100])
@pytest.mark.parametrize("n_features", [1, 10, 100])
@pytest.mark.parametrize("bounds", [10, 20, 30])
def test_splines_approx(n_samples, n_features, n_bins, bounds):

    # initialize function

    data_rng, *spline_rng = jax.random.split(KEY, 2)

    x = jax.random.normal(data_rng, shape=(n_samples, n_features))

    # initialize the parameters
    widths = jax.random.normal(spline_rng[0], shape=(1, n_features, n_bins))
    heights = jax.random.normal(spline_rng[1], shape=(1, n_features, n_bins))
    derivatives = jax.random.normal(spline_rng[2], shape=(1, n_features, n_bins))

    # create layer
    z, _ = rational_quadratic_spline(
        x, widths, heights, derivatives, bounds, inverse=False
    )

    # inverse transformation
    x_approx, _ = rational_quadratic_spline(
        z, widths, heights, derivatives, bounds, inverse=True
    )

    # checks
    chex.assert_tree_all_close(x, x_approx, atol=1e-5)
