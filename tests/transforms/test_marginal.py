# from functools import partial

# import chex
# import jax.numpy as np
# import numpy as onp

# from rbig_jax.transforms.gaussianize import get_gauss_params_hist
# from rbig_jax.transforms.marginal import (get_params_marginal,
#                                           marginal_transform)

# rng = onp.random.RandomState(123)
# X = rng.randn(1_000, 2)


# def test_marginal_params_shape():

#     # with loops
#     X_g_loop, params_loop = [], []
#     for iX in X.T:
#         iX_g, ix_params = get_gauss_params_hist(
#             iX, support_extension=10, precision=1000, alpha=1e-5
#         )
#         X_g_loop.append(iX_g)
#         params_loop.append(ix_params)

#     X_g_loop = np.vstack(X_g_loop).T

#     # with vmap
#     params_f = partial(
#         get_gauss_params_hist, support_extension=10, precision=1000, alpha=1e-5,
#     )
#     X_g, params = get_params_marginal(X, params_f)

#     chex.assert_equal_shape([X_g, X_g_loop])
#     chex.assert_equal(len(params_loop), len(params.support))
#     chex.assert_equal(len(params_loop), len(params.quantiles))
#     chex.assert_equal(len(params_loop), len(params.support_pdf))
#     chex.assert_equal(len(params_loop), len(params.empirical_pdf))


# def test_marginal_params_equal():

#     # with loops
#     X_g_loop, params_loop = [], []
#     for iX in X.T:
#         iX_g, ix_params = get_gauss_params_hist(
#             iX, support_extension=10, precision=1000, alpha=1e-5
#         )
#         X_g_loop.append(iX_g)
#         params_loop.append(ix_params)

#     X_g_loop = np.vstack(X_g_loop).T

#     # with vmap
#     params_f = partial(
#         get_gauss_params_hist, support_extension=10, precision=1000, alpha=1e-5,
#     )
#     X_g, params = get_params_marginal(X, params_f)

#     chex.assert_tree_all_close(X_g, X_g_loop)
#     params_loop = [
#         (iparam.support, iparam.empirical_pdf, iparam.quantiles, iparam.support_pdf)
#         for iparam in params_loop
#     ]
#     support, empirical_pdf, quantiles, support_pdf = map(np.vstack, zip(*params_loop))
#     chex.assert_tree_all_close(support, params.support)
#     chex.assert_tree_all_close(empirical_pdf, params.empirical_pdf)
#     chex.assert_tree_all_close(quantiles, params.quantiles)
#     chex.assert_tree_all_close(support_pdf, params.support_pdf)
