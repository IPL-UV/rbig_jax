# import chex
# import jax
# import jax.numpy as np
# import numpy as onp
# import objax
# import pytest
# from jax import random

# from rbig_jax.transforms.mixture import MixtureGaussianCDF, MixtureLogisticCDF

# seed = 123
# rng = onp.random.RandomState(123)
# generator = objax.random.Generator(123)


# @pytest.mark.parametrize("n_features", [1, 3, 10])
# @pytest.mark.parametrize("n_components", [1, 3, 10, 100])
# def test_mixturegausscdf_shape(n_features, n_components):

#     x = objax.random.normal((n_features,), generator=generator)

#     # create layer
#     model = MixtureGaussianCDF(n_features, n_components)

#     # forward transformation
#     z, log_abs_det = model(x)

#     # checks
#     chex.assert_equal_shape([z, x])
#     chex.assert_equal_shape([log_abs_det, x])
