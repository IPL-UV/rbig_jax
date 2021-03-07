import chex
import jax
import jax.numpy as np
import numpy as onp
import objax
import pytest

from rbig_jax.transforms.conv import Conv1x1

seed = 123
rng = onp.random.RandomState(123)
generator = objax.random.Generator(123)


@pytest.mark.parametrize("n_channels", [1, 3, 12])
@pytest.mark.parametrize("hw", [(1, 1), (5, 5), (12, 12)])
@pytest.mark.parametrize("n_samples", [1, 5, 10])
def test_conv1x1_shape(n_channels, hw, n_samples):

    x = objax.random.normal((n_samples, hw[0], hw[1], n_channels), generator=generator)
    # print(x.shape)
    # create layer
    model = Conv1x1(n_channels=n_channels)

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
def test_conv1x1_approx(n_channels, hw, n_samples):

    x = objax.random.normal((n_samples, hw[0], hw[1], n_channels), generator=generator)

    # create layer
    model = Conv1x1(n_channels=n_channels)

    # forward transformation
    z, log_abs_det = model(x)

    # inverse transformation
    x_approx = model.inverse(z)

    # checks
    chex.assert_tree_all_close(x, x_approx, atol=1e-5)
