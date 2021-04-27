import chex
import jax
import jax.numpy as np
import numpy as onp
import pytest

from rbig_jax.transforms.parametric.mixture.logistic import (
    MixtureLogisticCDF, mixture_logistic_cdf, mixture_logistic_cdf_vectorized,
    mixture_logistic_invcdf, mixture_logistic_invcdf_vectorized,
    mixture_logistic_log_pdf, mixture_logistic_log_pdf_vectorized)

seed = 123
rng = onp.random.RandomState(123)
# generator = objax.random.Generator(123)

KEY = jax.random.PRNGKey(seed)

# ====================================================
# MIXTURE GAUSSIAN CDF
# ====================================================


@pytest.mark.parametrize("n_features", [1, 3, 10])
@pytest.mark.parametrize("n_components", [1, 3, 10, 100])
def test_mixture_logistic_cdf_shape(n_features, n_components):

    params_rng, data_rng = jax.random.split(KEY, 2)

    x = jax.random.normal(data_rng, shape=(n_features,))

    # initialize mixture
    means = jax.random.normal(key=params_rng, shape=(n_features, n_components))
    log_scales = np.zeros((n_features, n_components))
    prior_logits = np.zeros((n_features, n_components))

    # transformation
    z = mixture_logistic_cdf(x, prior_logits, means, np.exp(log_scales))

    # checks
    chex.assert_equal_shape([z, x])


@pytest.mark.parametrize("n_samples", [1, 5, 10])
@pytest.mark.parametrize("n_features", [1, 5, 10])
@pytest.mark.parametrize("n_components", [1, 10])
def test_mixture_logistic_cdf_vmap_shape(n_samples, n_features, n_components):

    params_rng, data_rng = jax.random.split(KEY, 2)

    x = jax.random.normal(data_rng, shape=(n_samples, n_features))

    # initialize mixture
    means = jax.random.normal(key=params_rng, shape=(n_features, n_components))
    log_scales = np.zeros((n_features, n_components))
    prior_logits = np.zeros((n_features, n_components))

    # transformation
    z = mixture_logistic_cdf_vectorized(x, prior_logits, means, np.exp(log_scales))

    # checks
    chex.assert_equal_shape([z, x])


# ====================================================
# MIXTURE GAUSSIAN PDF
# ====================================================


@pytest.mark.parametrize("n_features", [1, 3, 10])
@pytest.mark.parametrize("n_components", [1, 3, 10, 100])
def test_mixture_logistic_pdf_shape(n_features, n_components):

    params_rng, data_rng = jax.random.split(KEY, 2)

    x = jax.random.normal(data_rng, shape=(n_features,))

    # initialize mixture
    means = jax.random.normal(key=params_rng, shape=(n_features, n_components))
    log_scales = np.zeros((n_features, n_components))
    prior_logits = np.zeros((n_features, n_components))

    # transformation
    z = mixture_logistic_log_pdf(x, prior_logits, means, np.exp(log_scales))

    # checks
    chex.assert_equal_shape([z, x])


@pytest.mark.parametrize("n_samples", [1, 5, 10])
@pytest.mark.parametrize("n_features", [1, 5, 10])
@pytest.mark.parametrize("n_components", [1, 10])
def test_mixture_logistic_pdf_vmap_shape(n_samples, n_features, n_components):

    params_rng, data_rng = jax.random.split(KEY, 2)

    x = jax.random.normal(data_rng, shape=(n_samples, n_features))

    # initialize mixture
    means = jax.random.normal(key=params_rng, shape=(n_features, n_components))
    log_scales = np.zeros((n_features, n_components))
    prior_logits = np.zeros((n_features, n_components))

    # transformation
    z = mixture_logistic_log_pdf_vectorized(x, prior_logits, means, np.exp(log_scales))
    # checks
    chex.assert_equal_shape([z, x])


# ====================================================
# INVERSE MIXTURE GAUSSIAN CDF
# ====================================================


@pytest.mark.parametrize("n_features", [1, 3, 10])
@pytest.mark.parametrize("n_components", [1, 3, 10, 100])
def test_logistic_log_invcdf_shape(n_features, n_components):

    params_rng, data_rng = jax.random.split(KEY, 2)

    z = jax.random.normal(data_rng, shape=(n_features,))

    # initialize mixture
    means = jax.random.normal(key=params_rng, shape=(n_features, n_components))
    log_scales = np.zeros((n_features, n_components))
    prior_logits = np.zeros((n_features, n_components))

    # transformation
    x = mixture_logistic_invcdf(z, prior_logits, means, np.exp(log_scales))

    # checks
    chex.assert_equal_shape([z, x])


@pytest.mark.parametrize("n_samples", [1, 5, 10])
@pytest.mark.parametrize("n_features", [1, 5, 10])
@pytest.mark.parametrize("n_components", [1, 10])
def test_mixture_logistic_invcdf_vmap_shape(n_samples, n_features, n_components):

    params_rng, data_rng = jax.random.split(KEY, 2)

    z = jax.random.normal(data_rng, shape=(n_samples, n_features))

    # initialize mixture
    means = jax.random.normal(key=params_rng, shape=(n_features, n_components))
    log_scales = np.zeros((n_features, n_components))
    prior_logits = np.zeros((n_features, n_components))

    # transformation
    x = mixture_logistic_invcdf_vectorized(z, prior_logits, means, np.exp(log_scales))

    # checks
    chex.assert_equal_shape([z, x])


# ====================================================
# MIXTURE GAUSSIAN CDF BIJECTOR
# ====================================================


@pytest.mark.parametrize("n_samples", [1, 5, 10])
@pytest.mark.parametrize("n_features", [1, 5, 10])
@pytest.mark.parametrize("n_components", [1, 10])
def test_mixture_logistic_cdf_bijector_shape(n_samples, n_features, n_components):

    params_rng, data_rng = jax.random.split(KEY, 2)

    x = jax.random.normal(data_rng, shape=(n_samples, n_features))

    # create layer
    init_func = MixtureLogisticCDF(n_components=n_components)

    # create layer
    params, forward_f, inverse_f = init_func(rng=params_rng, n_features=n_features)

    # forward transformation
    z, log_abs_det = forward_f(params, x)

    # checks
    chex.assert_equal_shape([z, x])
    chex.assert_shape(log_abs_det, (n_samples,))

    # forward transformation
    x_approx, log_abs_det = inverse_f(params, z)

    # checks
    chex.assert_equal_shape([x, x_approx])


@pytest.mark.parametrize("n_samples", [1, 5, 10])
@pytest.mark.parametrize("n_features", [1, 5, 10])
@pytest.mark.parametrize("n_components", [1, 10])
def test_mixture_logistic_cdf_bijector_approx(n_samples, n_features, n_components):

    params_rng, data_rng = jax.random.split(KEY, 2)

    x = jax.random.normal(data_rng, shape=(n_samples, n_features))

    # create layer
    init_func = MixtureLogisticCDF(n_components=n_components)

    # create layer
    params, forward_f, inverse_f = init_func(rng=params_rng, n_features=n_features)

    # forward transformation
    z, _ = forward_f(params, x)

    # forward transformation
    x_approx, _ = inverse_f(params, z)

    # checks
    chex.assert_tree_all_close(x, x_approx, rtol=1e-3)
