import collections
from typing import NamedTuple, Tuple
from chex._src.pytypes import PRNGKey
import jax.numpy as jnp
import jax
from chex import Array, dataclass
from distrax._src.bijectors.bijector import Bijector as distaxBijector
from rbig_jax.transforms.base import InitFunctions

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

    def init_params(inputs):
        outputs, params = f(inputs)
        return outputs, params

    def init_transform(inputs):
        outputs = f_slim(inputs)
        return outputs

    def init_bijector(inputs):
        outputs, params = init_params(inputs)
        bijector = Rotation(rotation=params.rotation,)
        return outputs, bijector

    return InitFunctions(
        init_params=init_params,
        init_bijector=init_bijector,
        init_transform=init_transform,
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

        outputs, params = f(rng, inputs)
        return outputs, params

    def init_transform(inputs, **kwargs):

        key_, rng = kwargs.get("rng", jax.random.split(key, 2))

        outputs = f_slim(inputs)
        return outputs

    def init_bijector(inputs, **kwargs):
        outputs, params = init_params(inputs, **kwargs)
        bijector = Rotation(rotation=params.rotation,)
        return outputs, bijector

    return InitFunctions(
        init_params=init_params,
        init_bijector=init_bijector,
        init_transform=init_transform,
    )


def get_pca_params(inputs: Array, return_params: bool = True) -> Array:

    # rotation
    rotation = compute_projection(inputs)
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


def compute_projection(X: jnp.ndarray) -> jnp.ndarray:
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
    _, _, VT = jnp.linalg.svd(X, full_matrices=False, compute_uv=True)

    return VT.T


def rot_forward_transform(X, params):
    return jnp.dot(X, params.rotation)


def rot_inverse_transform(X, params):
    return jnp.dot(X, params.rotation.T)


def rot_gradient_transform(X, params):
    return jnp.ones_like(X)
