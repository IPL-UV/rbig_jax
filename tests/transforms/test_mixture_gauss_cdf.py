import chex
import jax
import jax.numpy as np
import numpy as onp
import objax
import pytest
from jax import random

from rbig_jax.transforms.mixture.gaussian import (
    mixture_gaussian_cdf,
    mixture_gaussian_invcdf,
    mixture_gaussian_invcdf_vectorized,
    mixture_gaussian_cdf_vectorized,
    mixture_gaussian_log_pdf,
    mixture_gaussian_log_pdf_vectorized,
)

from rbig_jax.transforms.mixture import MixtureGaussianCDF

seed = 123
rng = onp.random.RandomState(123)
generator = objax.random.Generator(123)


# ====================================================
# MIXTURE GAUSSIAN CDF
# ====================================================


@pytest.mark.parametrize("n_features", [1, 3, 10])
@pytest.mark.parametrize("n_components", [1, 3, 10, 100])
def test_mixture_gaussian_cdf_shape(n_features, n_components):

    x = objax.random.normal((n_features,), generator=generator)

    prior_logits = objax.random.normal((n_features, n_components))
    mean = objax.random.normal((n_features, n_components))
    scale = objax.random.normal((n_features, n_components))

    # transformation
    z = mixture_gaussian_cdf(x, prior_logits, mean, scale)

    # checks
    chex.assert_equal_shape([z, x])


@pytest.mark.parametrize("n_samples", [1, 5, 10])
@pytest.mark.parametrize("n_features", [1, 5, 10])
@pytest.mark.parametrize("n_components", [1, 10])
def test_mixture_gaussian_cdf_vmap_shape(n_samples, n_features, n_components):

    x = objax.random.normal((n_samples, n_features,), generator=generator)

    prior_logits = objax.random.normal((n_features, n_components))
    mean = objax.random.normal((n_features, n_components))
    scale = objax.random.normal((n_features, n_components))

    # transformation
    z = mixture_gaussian_cdf_vectorized(x, prior_logits, mean, scale)

    # checks
    chex.assert_equal_shape([z, x])


# ====================================================
# INVERSE MIXTURE GAUSSIAN CDF
# ====================================================


@pytest.mark.parametrize("n_features", [1, 3, 10])
@pytest.mark.parametrize("n_components", [1, 3, 10, 100])
def test_gaussian_log_invcdf_shape(n_features, n_components):

    z = objax.random.uniform((n_features,), generator=generator)

    prior_logits = objax.random.normal((n_features, n_components))
    mean = objax.random.normal((n_features, n_components))
    scale = objax.random.normal((n_features, n_components))

    # transformation
    x = mixture_gaussian_invcdf(z, prior_logits, mean, scale)

    # checks
    chex.assert_equal_shape([z, x])


@pytest.mark.parametrize("n_samples", [1, 5, 10])
@pytest.mark.parametrize("n_features", [1, 5, 10])
@pytest.mark.parametrize("n_components", [1, 10])
def test_mixture_gaussian_invcdf_vmap_shape(n_samples, n_features, n_components):

    z = objax.random.uniform((n_samples, n_features,), generator=generator)

    prior_logits = objax.random.normal((n_features, n_components))
    mean = objax.random.normal((n_features, n_components))
    scale = objax.random.normal((n_features, n_components))

    # transformation
    x = mixture_gaussian_invcdf_vectorized(z, prior_logits, mean, scale)

    # checks
    chex.assert_equal_shape([z, x])


# ====================================================
# MIXTURE GAUSSIAN PDF
# ====================================================


@pytest.mark.parametrize("n_features", [1, 3, 10])
@pytest.mark.parametrize("n_components", [1, 3, 10, 100])
def test_mixture_gaussian_pdf_shape(n_features, n_components):

    x = objax.random.normal((n_features,), generator=generator)

    prior_logits = objax.random.normal((n_features, n_components))
    mean = objax.random.normal((n_features, n_components))
    scale = objax.random.normal((n_features, n_components))

    # transformation
    z = mixture_gaussian_log_pdf(x, prior_logits, mean, scale)

    # checks
    chex.assert_equal_shape([z, x])


@pytest.mark.parametrize("n_samples", [1, 5, 10])
@pytest.mark.parametrize("n_features", [1, 5, 10])
@pytest.mark.parametrize("n_components", [1, 10])
def test_mixture_gaussian_pdf_vmap_shape(n_samples, n_features, n_components):

    x = objax.random.normal((n_samples, n_features,), generator=generator)

    prior_logits = objax.random.normal((n_features, n_components))
    mean = objax.random.normal((n_features, n_components))
    scale = objax.random.normal((n_features, n_components))

    # transformation
    z = mixture_gaussian_log_pdf_vectorized(x, prior_logits, mean, scale)

    # checks
    chex.assert_equal_shape([z, x])


# ====================================================
# MIXTURE GAUSSIAN CDF BIJECTOR
# ====================================================


@pytest.mark.parametrize("n_samples", [1, 5, 10])
@pytest.mark.parametrize("n_features", [1, 5, 10])
@pytest.mark.parametrize("n_components", [1, 10])
def test_mixturegausscdf_bijector_shape(n_samples, n_features, n_components):

    x = objax.random.normal((n_samples, n_features,), generator=generator)

    # create layer
    model = MixtureGaussianCDF(n_features, n_components)

    # forward transformation
    z, log_abs_det = model(x)

    # checks
    chex.assert_equal_shape([z, x])
    chex.assert_shape(log_abs_det, (n_samples,))

    # forward transformation
    x_approx = model.inverse(z)

    # checks
    chex.assert_equal_shape([x, x_approx])


@pytest.mark.parametrize("n_samples", [1, 5, 10])
@pytest.mark.parametrize("n_features", [1, 5, 10])
@pytest.mark.parametrize("n_components", [1, 10])
def test_mixturegausscdf_bijector_approx(n_samples, n_features, n_components):

    x = objax.random.normal((n_samples, n_features,), generator=generator)

    # create layer
    model = MixtureGaussianCDF(n_features, n_components)

    # forward transformation
    z, _ = model(x)

    # forward transformation
    x_approx = model.inverse(z)

    # checks
    chex.assert_tree_all_close(x, x_approx, rtol=1e-3)
