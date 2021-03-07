from typing import Tuple

import jax.numpy as np
import numpy as onp
from chex import Array


def get_data(
    N: int = 30,
    input_noise: float = 0.15,
    output_noise: float = 0.15,
    N_test: int = 400,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    onp.random.seed(0)
    X = np.linspace(-1, 1, N)
    Y = X + 0.2 * np.power(X, 3.0) + 0.5 * np.power(0.5 + X, 2.0) * np.sin(4.0 * X)
    Y += output_noise * onp.random.randn(N)
    Y -= np.mean(Y)
    Y /= np.std(Y)

    X += input_noise * onp.random.randn(N)

    assert X.shape == (N,)
    assert Y.shape == (N,)

    X_test = np.linspace(-1.2, 1.2, N_test)

    return X[:, None], Y[:, None], X_test[:, None]


def get_classic(n_samples=10_000, seed=123):
    rng = onp.random.RandomState(seed=seed)
    x = onp.abs(2 * rng.randn(1, n_samples))
    y = onp.sin(x) + 0.25 * rng.randn(1, n_samples)
    return onp.vstack((x, y)).T


def generate_2d_grid(data: Array, n_grid: int = 1_000, buffer: float = 0.01) -> Array:

    xline = np.linspace(data[:, 0].min() - buffer, data[:, 0].max() + buffer, n_grid)
    yline = np.linspace(data[:, 1].min() - buffer, data[:, 1].max() + buffer, n_grid)
    xgrid, ygrid = np.meshgrid(xline, yline)
    xyinput = np.concatenate([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], axis=1)
    return xyinput
