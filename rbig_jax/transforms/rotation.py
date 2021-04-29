import collections
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from chex import Array, dataclass
from chex._src.pytypes import PRNGKey
from distrax._src.bijectors.bijector import Bijector as distaxBijector

from rbig_jax.transforms.base import InitFunctions, InitLayersFunctions

RotParams = collections.namedtuple("Params", ["rotation"])


class RotationParams(NamedTuple):
    rotation: Array


class Rotation(distaxBijector):
    def __init__(self, rotation: Array):
        self.rotation = rotation

    def forward_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:

        # clip inputs within boundary
        outputs = jnp.dot(inputs, self.rotation)

        # we know it's zero
        logabsdet = jnp.zeros_like(outputs)

        return outputs, logabsdet

    def forward(self, inputs: Array) -> Array:

        # clip inputs within boundary
        outputs = jnp.dot(inputs, self.rotation)

        return outputs

    def inverse_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:

        outputs = jnp.dot(inputs, self.rotation.T)

        # we know it's zero
        logabsdet = jnp.zeros_like(outputs)

        return outputs, logabsdet

    def inverse(self, inputs: Array) -> Array:

        # clip inputs within boundary
        outputs = jnp.dot(inputs, self.rotation.T)

        return outputs

    def forward_log_det_jacobian(self, inputs: Array) -> Array:

        # we know it's zero
        logabsdet = jnp.zeros_like(inputs)

        return logabsdet

    def inverse_log_det_jacobian(self, inputs: Array) -> Array:

        # we know it's zero
        logabsdet = jnp.zeros_like(inputs)

        return logabsdet


def InitPCARotation(jitted=False):
    # create marginal functions

    f = jax.partial(get_pca_params, return_params=True,)

    f_slim = jax.partial(get_pca_params, return_params=False,)

    if jitted:
        f = jax.jit(f)
        f_slim = jax.jit(f_slim)

    def init_params(inputs, **kwargs):
        _, params = f(inputs)
        return params

    def params_and_transform(inputs, **kwargs):

        outputs, params = f(inputs)
        return outputs, params

    def transform(inputs, **kwargs):

        outputs = f_slim(inputs)
        return outputs

    def bijector(inputs, **kwargs):
        params = init_params(inputs, **kwargs)
        bijector = Rotation(rotation=params.rotation,)
        return bijector

    def bijector_and_transform(inputs, **kwargs):
        outputs, params = params_and_transform(inputs, **kwargs)
        bijector = Rotation(rotation=params.rotation,)
        return outputs, bijector

    return InitLayersFunctions(
        bijector=bijector,
        bijector_and_transform=bijector_and_transform,
        transform=transform,
        params=init_params,
        params_and_transform=params_and_transform,
    )


def InitRandomRotation(rng: PRNGKey, jitted=False):
    # create marginal functions
    key = rng
    f = jax.partial(get_random_rotation, return_params=True,)

    f_slim = jax.partial(get_random_rotation, return_params=False,)

    if jitted:
        f = jax.jit(f)
        f_slim = jax.jit(f_slim)

    def init_params(inputs, **kwargs):

        key_, rng = kwargs.get("rng", jax.random.split(key, 2))

        _, params = f(rng, inputs)
        return params

    def params_and_transform(inputs, **kwargs):

        key_, rng = kwargs.get("rng", jax.random.split(key, 2))

        outputs, params = f(rng, inputs)
        return outputs, params

    def transform(inputs, **kwargs):

        key_, rng = kwargs.get("rng", jax.random.split(key, 2))

        outputs = f_slim(inputs)
        return outputs

    def bijector(inputs, **kwargs):
        params = init_params(inputs, **kwargs)
        bijector = Rotation(rotation=params.rotation,)
        return bijector

    def bijector_and_transform(inputs, **kwargs):
        print(inputs.shape)
        outputs, params = params_and_transform(inputs, **kwargs)
        bijector = Rotation(rotation=params.rotation,)
        return outputs, bijector

    return InitLayersFunctions(
        bijector=bijector,
        bijector_and_transform=bijector_and_transform,
        transform=transform,
        params=init_params,
        params_and_transform=params_and_transform,
    )


def get_pca_params(
    inputs: Array, full_matrices: bool = False, return_params: bool = True
) -> Array:

    # rotation
    rotation = compute_projection(inputs, full_matrices=full_matrices)
    outputs = jnp.dot(inputs, rotation)

    if return_params:
        return outputs, RotationParams(rotation=rotation)
    else:
        return outputs


def get_random_rotation(
    rng: PRNGKey, inputs: Array, return_params: bool = True
) -> Array:

    # compute orthogonal matrix
    rotation = jax.nn.initializers.orthogonal()(
        key=rng, shape=(inputs.shape[1], inputs.shape[1])
    )
    # compute dot product
    outputs = jnp.dot(inputs, rotation)

    if return_params:
        return outputs, RotationParams(rotation=rotation)
    else:
        return outputs


def compute_projection(X: jnp.ndarray, full_matrices: bool = False) -> jnp.ndarray:
    """Compute PCA projection matrix
    Using SVD, this computes the PCA components for
    a dataset X and computes the projection matrix
    needed to do the PCA decomposition.

    Parameters
    ----------
    X : jnp.ndarray, (n_samples, n_features)
        the data to calculate to PCA projection matrix
    
    Returns
    -------
    VT : jnp.ndarray, (n_features, n_features)
        the projection matrix (V.T) for the PCA decomposition

    Notes
    -----
    Can find the original implementation here:
    https://bit.ly/2EBDV9o
    """

    # center the data
    X = X - jnp.mean(X, axis=0)

    # Compute SVD
    _, _, VT = jnp.linalg.svd(X, full_matrices=full_matrices, compute_uv=True)

    return VT.T


def rot_forward_transform(X, params):
    return jnp.dot(X, params.rotation)


def rot_inverse_transform(X, params):
    return jnp.dot(X, params.rotation.T)


def rot_gradient_transform(X, params):
    return jnp.ones_like(X)
