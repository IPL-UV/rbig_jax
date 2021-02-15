import jax
import jax.numpy as np
import numpy as onp
import pytest
from jax import random

from rbig_jax.utils import (
    searchsorted,
    bisection_body,
    bisection_search,
    bisection_search_vmap,
    BisectionState,
)
import chex

rng = onp.random.RandomState(123)


@pytest.mark.parametrize("shape", [(1,), (3,), (3, 6, 6)])
def test_bisection_body_shapes(shape):

    x = rng.randn(*shape)

    # create function
    f = jax.nn.sigmoid

    # true forward solution
    y = f(x)

    # initialize state
    x_approx = np.zeros_like(y)
    init_ub = np.ones_like(y) + 1000.0
    init_lb = np.ones_like(y) - 1000.0
    init_diff = np.ones_like(y) * 10.0

    state_init = BisectionState(
        x=y.squeeze(),
        current_x=f(x_approx).squeeze(),
        current_y=x_approx.squeeze(),
        lower_bound=init_lb.squeeze(),
        upper_bound=init_ub.squeeze(),
        diff=init_diff.squeeze(),
        iteration=0,
    )

    # bisection body forward
    state_next = bisection_body(f, state_init)

    # ==============
    # CHECK STATES
    # ==============

    # input data
    chex.assert_equal_shape([state_init.x, state_next.x])
    # forward solution
    chex.assert_equal_shape([state_init.current_x, state_next.current_x])
    # solution
    chex.assert_equal_shape([state_init.current_y, state_next.current_y])
    # lower bound
    chex.assert_equal_shape([state_init.diff, state_next.diff])
    # upper bound
    chex.assert_equal_shape([state_init.upper_bound, state_next.upper_bound])
    # difference
    chex.assert_equal_shape([state_init.lower_bound, state_next.lower_bound])


@pytest.mark.parametrize("shape", [(1,), (3,), (3, 6, 6)])
def test_bisection_search_types(shape):

    x = rng.randn(*shape)
    # create function
    f = jax.nn.sigmoid

    # true forward solution
    y = f(x)

    # initialize state
    # initialize state
    x_approx = np.zeros_like(y)
    init_ub = np.ones_like(y) + 1000.0
    init_lb = np.ones_like(y) - 1000.0

    x_approx = bisection_search(f, y, init_lb, init_ub, atol=1e-8, max_iters=1_000)
    chex.assert_type(x_approx, x.dtype)


@pytest.mark.parametrize("shape", [(1,), (3,), (3, 6, 6)])
def test_bisection_search_shape(shape):

    x = np.array(rng.randn(*shape))

    # create function
    f = jax.nn.sigmoid

    # true forward solution
    y = f(x)

    # initialize state
    # initialize state
    x_approx = np.zeros_like(y)
    init_ub = np.ones_like(y) + 1000.0
    init_lb = np.ones_like(y) - 1000.0

    x_approx = bisection_search(f, y, init_lb, init_ub, atol=1e-8, max_iters=1_000)

    chex.assert_equal_shape([x_approx, x])


@pytest.mark.parametrize("shape", [(10, 1,), (10, 3,), (10, 3, 6, 6)])
def test_bisection_search_vmap_shape(shape):

    x = np.array(rng.randn(*shape))

    # create function
    f = jax.nn.sigmoid

    # true forward solution
    y = f(x)

    # initialize state
    # initialize state
    init_ub = np.ones_like(y[0]) + 1000.0
    init_lb = np.ones_like(y[0]) - 1000.0
    x_approx = bisection_search_vmap(f, y, init_lb, init_ub, 1e-8, 1_000)

    chex.assert_equal_shape([x_approx, x])


@pytest.mark.parametrize("shape", [(1,), (3,), (10,), (50,), (100,)])
@pytest.mark.parametrize("search_tol", [1e-10])
@pytest.mark.parametrize("bisection_tol", [1e-5])
def test_bisection_search(shape, search_tol, bisection_tol):

    x = rng.randn(*shape)
    # create function
    f = jax.nn.sigmoid

    # true forward solution
    y = f(x)

    # initialize state
    # initialize state
    x_approx = np.zeros_like(y)
    init_ub = np.ones_like(y) + 1000.0
    init_lb = np.ones_like(y) - 1000.0

    x_approx = bisection_search(
        f, y, init_lb, init_ub, atol=search_tol, max_iters=1_000
    )

    chex.assert_tree_all_close(x_approx, x, atol=bisection_tol)


@pytest.mark.parametrize(
    "shape", [(10, 1,), (10, 3,), (10, 10,), (10, 50,), (10, 100,)]
)
@pytest.mark.parametrize("search_tol", [1e-10])
@pytest.mark.parametrize("bisection_tol", [1e-5])
def test_bisection_search_vmap_search(shape, search_tol, bisection_tol):

    x = np.array(rng.randn(*shape))

    # create function
    f = jax.nn.sigmoid

    # true forward solution
    y = f(x)

    # initialize state
    # initialize state
    init_ub = np.ones_like(y[0]) + 1000.0
    init_lb = np.ones_like(y[0]) - 1000.0
    x_approx = bisection_search_vmap(f, y, init_lb, init_ub, search_tol, 1_000)

    chex.assert_tree_all_close(x_approx, x, atol=bisection_tol)


def test_searchsorted():

    # create 1d array
    bin_locations = np.linspace(0, 1, 10)

    left_boundaries = bin_locations[:-1]
    right_boundaries = bin_locations[:-1] + 0.1
    mid_points = bin_locations[:-1] + 0.05
    print("Bins:", bin_locations.shape)

    for inputs in [left_boundaries, right_boundaries, mid_points]:
        idx = searchsorted(bin_locations[None, :], left_boundaries)

        onp.testing.assert_array_equal(onp.array(idx), onp.arange(0, 9))


def test_searchsorted_arbitrary_shape():
    shape = [2, 3, 4]
    bin_locations = np.linspace(0, 1, 10)
    bin_locations = np.tile(bin_locations, reps=(*shape, 1))

    # initialize random
    key = random.PRNGKey(0)
    inputs = random.uniform(key, shape=(*shape,))
    print(inputs.shape, bin_locations.shape)

    idx = searchsorted(bin_locations, inputs)

    onp.testing.assert_equal(inputs.shape, idx.shape)


def test_searchsorted_double():

    bin_locations = np.linspace(0, 1, 10)
    bin_locations = np.tile(bin_locations, reps=(2, 1))
