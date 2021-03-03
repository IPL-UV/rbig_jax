import chex
import jax
import jax.numpy as np
import numpy as onp
import objax
import pytest

from rbig_jax.transforms.conv import Conv1x1Householder

seed = 123
rng = onp.random.RandomState(123)
generator = objax.random.Generator(123)


@pytest.mark.parametrize("n_channels", [1, 3, 12])
@pytest.mark.parametrize("hw", [(1, 1), (5, 5), (12, 12)])
@pytest.mark.parametrize("n_samples", [1, 5, 10])
@pytest.mark.parametrize("n_reflections", [1, 2, 10])
def test_conv1x1ortho_shape(n_channels, hw, n_samples, n_reflections):

    x = objax.random.normal((n_samples, hw[0], hw[1], n_channels), generator=generator)
    # print(x.shape)
    # create layer
    model = Conv1x1Householder(n_channels=n_channels, n_reflections=n_reflections)

    # forward transformation
    z, log_abs_det = model(x)

    # print(z.shape, log_abs_det.shape)

    # checks
    chex.assert_equal_shape([z, x])
    chex.assert_shape(np.atleast_1d(log_abs_det), (n_samples,))

    # forward transformation
    z = model.transform(x)

    # checks
    chex.assert_equal_shape([z, x])

    # inverse transformation
    x_approx = model.inverse(z)

    # checks
    chex.assert_equal_shape([x_approx, x])


@pytest.mark.parametrize("n_channels", [1, 3, 12])
@pytest.mark.parametrize("hw", [(1, 1), (5, 5), (12, 12)])
@pytest.mark.parametrize("n_samples", [1, 5, 10])
@pytest.mark.parametrize("n_reflections", [1, 2, 10])
def test_conv1x1ortho_approx(n_channels, hw, n_samples, n_reflections):

    x = objax.random.normal((n_samples, hw[0], hw[1], n_channels), generator=generator)

    # create layer
    model = Conv1x1Householder(n_channels=n_channels, n_reflections=n_reflections)

    # forward transformation
    z, log_abs_det = model(x)

    # inverse transformation
    x_approx = model.inverse(z)

    # checks
    chex.assert_tree_all_close(x, x_approx, atol=1e-5)
