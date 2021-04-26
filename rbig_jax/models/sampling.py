from rbig_jax.models.gaussflow import GaussianizationFlow
from chex import dataclass, Array
from rbig_jax.models.gaussflow import GaussianizationFlow
from rbig_jax.transforms.base import Bijector
from typing import Iterable, Callable, Tuple
from distrax._src.distributions.distribution import Distribution
import jax.numpy as jnp


@dataclass
class GFSampling(GaussianizationFlow):
    bijectors: Iterable[Bijector]
    base_dist: Distribution
    ibijectors: Iterable[Bijector]

    def inverse_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:

        outputs = inputs
        total_logabsdet = jnp.zeros_like(outputs)
        # total_logabsdet = jnp.expand_dims(total_logabsdet, axis=1)
        for ibijector in self.ibijectors:

            outputs, logabsdet = ibijector.inverse_and_log_det(outputs)
            total_logabsdet += logabsdet  # sum_last(logabsdet, ndims=logabsdet.ndim)

        return outputs, total_logabsdet

    def inverse(self, inputs: Array) -> Array:

        outputs = inputs
        for ibijector in self.ibijectors:
            outputs = ibijector.inverse(outputs)

        return outputs

    def inverse_log_det_jacobian(self, inputs: Array) -> Tuple[Array, Array]:

        outputs = inputs
        total_logabsdet = jnp.zeros_like(outputs)
        # total_logabsdet = jnp.expand_dims(total_logabsdet, axis=1)
        for ibijector in self.ibijectors:

            outputs, logabsdet = ibijector.inverse_and_log_det(outputs)
            total_logabsdet += logabsdet  # sum_last(logabsdet, ndims=logabsdet.ndim)

        return total_logabsdet

    @property
    def name(self) -> str:
        """Name of the bijector."""
        return self.__class__.__name__

    def sample(self, seed: int, n_samples: int, jitted: bool = False):
        # generate Gaussian samples
        X_g_samples = self.base_dist.sample(seed=seed, sample_shape=n_samples)
        # # inverse transformation
        if jitted:
            f = self.inverse
            return f(X_g_samples)
        else:
            return self.inverse(X_g_samples)


@dataclass
class InverseSampler:
    marginal_gauss: Iterable[Callable]

    def inverse_log_det_jacobian(self, inputs):

        return jnp.zeros_like(inputs)

    def inverse(self, inputs):
        outputs = inputs
        for itransform in self.marginal_gauss:
            outputs = itransform.transform(outputs)

        return outputs

    def inverse_and_log_det(self, inputs):
        outputs = inputs
        for itransform in self.marginal_gauss:
            outputs = itransform.transform(inputs)

        return outputs, jnp.zeros_like(outputs)

    @property
    def name(self) -> str:
        """Name of the bijector."""
        return self.__class__.__name__


from copy import deepcopy
import tqdm


def init_gf_inverse_sampler(model, marginal_init_f: Iterable[Callable]):

    # create list of inverse bijectors
    inverse_bijectors = deepcopy(model.bijectors)

    inverse_bijector = InverseSampler(marginal_gauss=marginal_init_f)

    # insert the marginal transformation inside
    new_ibijectors = []
    with tqdm.tqdm(inverse_bijectors[::-1]) as pbar:
        for ib in pbar:
            pbar.set_description(f"Layer: {ib.name}")
            if ib.name in ["InverseGaussCDF", "Logit"]:
                new_ibijectors.append(inverse_bijector)
            new_ibijectors.append(ib)

    rbig_model_sampler = GFSampling(
        bijectors=model.bijectors, ibijectors=new_ibijectors, base_dist=model.base_dist,
    )
    return rbig_model_sampler
