import jax.numpy as np
import numpy as onp
import chex
from rbig_jax.transforms.logit import Logit

rng = onp.random.RandomState(123)


def test_hist_params_transform():

    X_u = rng.uniform(100)

    model = Logit()

    X_g = model(X_u)

    X_approx = model.inverse(X_g)

    chex.assert_tree_all_close(X_u, X_approx)
