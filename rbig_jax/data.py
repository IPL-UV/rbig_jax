from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np
from chex import Array
from sklearn import datasets
from torch.utils.data import DataLoader, Dataset

# import torch.multiprocessing as multiprocessing

# multiprocessing.set_start_method("spawn")
# class DensityDataset:
#     def __init__(self, data, dtype=jnp.float32):
#         self.data = data
#         self.dtype = dtype

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx: int) -> Array:
#         data = self.data[idx]
#         return jnp.array(data, dtype=self.dtype)


class DensityDataset(Dataset):
    def __init__(self, n_samples: int = 10_000, noise: float = 0.1, seed: int = 123):
        self.n_samples = n_samples
        self.seed = seed
        self.noise = noise
        self.reset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Array:
        data = self.data[idx]
        return data

    def reset(self):
        self._create_data()

    def _create_data(self):
        raise NotImplementedError


class GenericDataset(DensityDataset):
    def __init__(self, data):
        self.data = data


def collate_fn(batch):
    if isinstance(batch[0], jnp.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        return type(batch[0])(collate_fn(samples) for samples in zip(*batch))
    else:
        return jnp.asarray(batch)


def get_data(
    N: int = 30,
    input_noise: float = 0.15,
    output_noise: float = 0.15,
    N_test: int = 400,
) -> Tuple[np.ndarray, jnp.ndarray, jnp.ndarray]:
    np.random.seed(0)
    X = jnp.linspace(-1, 1, N)
    Y = X + 0.2 * jnp.power(X, 3.0) + 0.5 * jnp.power(0.5 + X, 2.0) * jnp.sin(4.0 * X)
    Y += output_noise * np.random.randn(N)
    Y -= jnp.mean(Y)
    Y /= jnp.std(Y)

    X += input_noise * np.random.randn(N)

    assert X.shape == (N,)
    assert Y.shape == (N,)

    X_test = jnp.linspace(-1.2, 1.2, N_test)

    return X[:, None], Y[:, None], X_test[:, None]


class SCurveDataset(DensityDataset):
    def _create_data(self):
        data, _ = datasets.make_s_curve(
            n_samples=self.n_samples, noise=self.noise, random_state=self.seed
        )
        data = data[:, [0, 2]]
        self.data = data


class BlobsDataset(DensityDataset):
    def _create_data(self):
        data, _ = datasets.make_blobs(n_samples=self.n_samples, random_state=self.seed)
        self.data = data


class MoonsDataset(DensityDataset):
    def _create_data(self):
        data, _ = datasets.make_moons(
            n_samples=self.n_samples, noise=self.noise, random_state=self.seed
        )
        self.data = data


class SwissRollDataset(DensityDataset):
    def _create_data(self):
        data, _ = datasets.make_swiss_roll(
            n_samples=self.n_samples, noise=self.noise, random_state=self.seed
        )
        data = data[:, [0, 2]]
        self.data = data


class NoisySineDataset(DensityDataset):
    def _create_data(self):

        rng = np.random.RandomState(seed=self.seed)
        x = np.abs(2 * rng.randn(1, self.n_samples))
        y = np.sin(x) + 0.25 * rng.randn(1, self.n_samples)
        self.data = np.vstack((x, y)).T


def get_classic(n_samples=10_000, seed=123):
    rng = np.random.RandomState(seed=seed)
    x = np.abs(2 * rng.randn(1, n_samples))
    y = np.sin(x) + 0.25 * rng.randn(1, n_samples)
    return np.vstack((x, y)).T


def generate_2d_grid(data: Array, n_grid: int = 1_000, buffer: float = 0.01) -> Array:

    xline = jnp.linspace(data[:, 0].min() - buffer, data[:, 0].max() + buffer, n_grid)
    yline = jnp.linspace(data[:, 1].min() - buffer, data[:, 1].max() + buffer, n_grid)
    xgrid, ygrid = jnp.meshgrid(xline, yline)
    xyinput = jnp.concatenate([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], axis=1)
    return xyinput
