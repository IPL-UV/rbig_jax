import chex
import jax
import jax.numpy as np
import numpy as onp
import objax
import pytest
from jax import random
from rbig_jax.transforms.kde import (
    broadcast_kde_pdf,
    broadcast_kde_cdf,
    normalization_factor,
)
from rbig_jax.utils import get_domain_extension

seed = 123
rng = onp.random.RandomState(123)
generator = objax.random.Generator(123)


# ====================================================
# MIXTURE GAUSSIAN CDF
# ====================================================


@pytest.mark.parametrize("n_samples", [10, 100, 1_000])
def test_broadcast_kde_pdf_shape(n_samples):

    bw = 0.1
    precision = 10

    x = objax.random.normal((n_samples,), generator=generator)
    lb, ub = get_domain_extension(x, 10)
    support = np.linspace(lb, ub, precision)

    pdf_support = broadcast_kde_pdf(support, x, bw)

    # checks
    chex.assert_shape(support, (precision,))
    chex.assert_shape(pdf_support, (precision,))


@pytest.mark.parametrize("n_samples", [10, 100, 1_000])
def test_normalization_factor_shape(n_samples):

    bw = 0.1

    x = objax.random.normal((n_samples,), generator=generator)

    factor = normalization_factor(x, bw)

    # checks
    chex.assert_shape(factor, ())


@pytest.mark.parametrize("n_samples", [10, 100, 1_000])
def test_broadcast_kde_cdf_shape(n_samples):

    bw = 0.1
    precision = 10

    x = objax.random.normal((n_samples,), generator=generator)
    lb, ub = get_domain_extension(x, 10)
    support = np.linspace(lb, ub, precision)

    factor = normalization_factor(x, bw)

    quantiles = broadcast_kde_cdf(support, x, factor)

    # checks
    chex.assert_shape(quantiles, (precision,))
