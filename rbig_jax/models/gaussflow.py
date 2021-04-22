import jax.numpy as jnp
from typing import Iterable
from chex import Array, dataclass
from rbig_jax.transforms.base import BijectorChain, Bijector
from distrax._src.distributions.distribution import Distribution
from rbig_jax.transforms.parametric.mixture.logistic import InitMixtureLogisticCDF
from rbig_jax.transforms.logit import InitLogitTransform
from rbig_jax.transforms.parametric.householder import InitHouseHolder
import jax
import corner
from distrax._src.distributions.normal import Normal


@dataclass
class GaussianizationFlow(BijectorChain):
    bijectors: Iterable[Bijector]
    base_dist: Distribution

    def score_samples(self, inputs):

        # forward propagation
        z, log_det = self.forward_and_log_det(inputs)

        # calculate latent probability
        latent_prob = self.base_dist.log_prob(z)

        # calculate log probability
        log_prob = latent_prob.sum(axis=1) + log_det.sum(axis=1)

        return log_prob

    def score(self, inputs):
        return -jnp.mean(self.score_samples(inputs))

    def sample(self, seed: int, n_samples: int):
        # generate Gaussian samples
        X_g_samples = self.base_dist.sample(seed=seed, sample_shape=n_samples)
        # # inverse transformation
        return self.inverse(X_g_samples)


def init_default_gf_model(
    shape: tuple,
    X: Array = None,
    n_layers: int = 4,
    n_components: int = 20,
    init_mixcdf: str = "gmm",
    init_rotation: str = "pca",
    n_reflections: int = 10,
    plot_layers: bool = False,
    plot_blocks: bool = False,
):

    n_features = shape[0]
    KEY = jax.random.PRNGKey(123)

    init_mixcdf_f = InitMixtureLogisticCDF(
        n_components=n_components, init_method=init_mixcdf
    )

    # Logit Transform
    init_logit_f = InitLogitTransform()

    # ==================
    # HouseHolder Transform
    # ======================
    n_reflections = n_reflections
    # initialize init function
    init_hh_f = InitHouseHolder(n_reflections=n_reflections, method=init_rotation)

    rng, *layer_rngs = jax.random.split(KEY, num=n_layers + 1)

    bijectors = []

    X_g = X.copy()

    if plot_blocks:
        fig = corner.corner(X_g, color="red", hist_bin_factor=2)

    for irng in layer_rngs:

        # ======================
        # MIXTURECDF
        # ======================
        # create keys for all inits
        irng, icdf_rng = jax.random.split(irng, 2)

        # intialize bijector and transformation
        X_g, layer = init_mixcdf_f.bijector_and_transform(
            inputs=X_g, rng=icdf_rng, n_features=n_features
        )

        # plot data
        if plot_layers and plot_blocks:
            fig = corner.corner(X_g, color="red", hist_bin_factor=2)

        # add bijector to list
        bijectors.append(layer)

        # ======================
        # LOGIT
        # ======================

        # intialize bijector and transformation
        X_g, layer = init_logit_f.bijector_and_transform(inputs=X_g)

        bijectors.append(layer)

        # plot data
        if plot_layers and plot_blocks:
            fig = corner.corner(X_g, color="red", hist_bin_factor=2)

        # ======================
        # HOUSEHOLDER
        # ======================

        # create keys for all inits
        irng, hh_rng = jax.random.split(irng, 2)

        # intialize bijector and transformation
        X_g, layer = init_hh_f.bijector_and_transform(
            inputs=X_g, rng=hh_rng, n_features=n_features
        )

        bijectors.append(layer)

        # plot data
        if plot_blocks:
            print(plot_layers)
            fig = corner.corner(X_g, color="red", hist_bin_factor=2)

    # create base dist
    base_dist = Normal(jnp.zeros((n_features,)), jnp.ones((n_features,)))

    # create flow model
    gf_model = GaussianizationFlow(base_dist=base_dist, bijectors=bijectors)
    return gf_model
