# import chex
# import jax.numpy as np
# import numpy as onp

# from rbig_jax.transforms.histogram import get_hist_params
# from rbig_jax.transforms.kde import get_kde_params
# from rbig_jax.transforms.uniformize import (
#     uniformize_transform,
#     uniformize_inverse,
# )

# # TODO: Think about a test for the uniformization gradient

# rng = onp.random.RandomState(123)
# X = rng.randn(100)


# def test_hist_uniformize():

#     X_u, params = get_hist_params(X, support_extension=10, precision=1000, alpha=1e-5)

#     X_u_trans = uniformize_transform(X, params)

#     chex.assert_tree_all_close(X_u, X_u_trans)


# def test_kde_uniformize():

#     X_u, params = get_kde_params(X, support_extension=10, precision=1000)

#     X_u_trans = uniformize_transform(X, params)

#     chex.assert_tree_all_close(X_u, X_u_trans)


# def test_hist_uniformize_bounds():

#     X_u, params = get_hist_params(X, support_extension=10, precision=1000, alpha=1e-5)

#     X_u_trans = uniformize_transform(X, params)

#     assert X_u_trans.max() <= 1.0
#     assert X_u_trans.min() >= 0.0


# def test_kde_uniformize_bounds():

#     X_u, params = get_kde_params(X, support_extension=10, precision=1000)

#     X_u_trans = uniformize_transform(X, params)

#     assert X_u_trans.max() <= 1.0
#     assert X_u_trans.min() >= 0.0


# def test_hist_inv_uniformize():

#     X_u, params = get_hist_params(X, support_extension=10, precision=1000, alpha=1e-5)

#     X_approx = uniformize_inverse(X_u, params)

#     chex.assert_tree_all_close(X, X_approx, atol=1e-4)


# def test_kde_inv_uniformize():

#     X_u, params = get_kde_params(X, support_extension=10, precision=1000)

#     X_approx = uniformize_inverse(X_u, params)

#     chex.assert_tree_all_close(X, X_approx, atol=1e-4)
