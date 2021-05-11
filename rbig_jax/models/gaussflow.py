import itertools
from typing import Iterable

import corner
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from chex import Array, dataclass
from distrax._src.distributions.distribution import Distribution
from distrax._src.distributions.normal import Normal

from rbig_jax.transforms.base import Bijector, BijectorChain
from rbig_jax.transforms.inversecdf import InitInverseGaussCDF, InitGaussCDF
from rbig_jax.transforms.logit import InitLogitTransform, InitSigmoidTransform
from rbig_jax.transforms.parametric.householder import InitHouseHolder
from rbig_jax.transforms.parametric.mixture.gaussian import InitMixtureGaussianCDF
from rbig_jax.transforms.parametric.mixture.logistic import InitMixtureLogisticCDF
from rbig_jax.transforms.parametric.splines import InitPiecewiseRationalQuadraticCDF


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
    n_blocks: int = 4,
    n_components: int = 20,
    mixture: str = "logistic",
    init_mixcdf: str = "gmm",
    init_rotation: str = "pca",
    inverse_cdf: str = "logistic",
    n_reflections: int = 10,
):

    n_features = shape[0]
    rng = jax.random.PRNGKey(42)
    # rng, _ = jax.random.split(jax.random.PRNGKey(123), 2)

    if mixture == "logistic":
        init_mixcdf_f = InitMixtureLogisticCDF(
            n_components=n_components, init_method=init_mixcdf
        )

    elif mixture == "gaussian":
        init_mixcdf_f = InitMixtureGaussianCDF(
            n_components=n_components, init_method=init_mixcdf
        )
    else:
        raise ValueError(f"Unrecognized mixture dist: {mixture}")

    if inverse_cdf == "logistic":
        # Logit Transform
        init_icdf_f = InitLogitTransform()
    elif inverse_cdf == "gaussian":
        init_icdf_f = InitInverseGaussCDF()
    else:
        raise ValueError(f"Unrecognized inverse cdf function: {inverse_cdf}")
    # =====================
    # HouseHolder Transform
    # ======================
    n_reflections = n_reflections
    # initialize init function
    init_hh_f = InitHouseHolder(n_reflections=n_reflections, method=init_rotation)

    block_rngs = jax.random.split(rng, num=n_blocks)
    # rng = jax.random.split(jax.random.PRNGKey(42), n_blocks)
    # block_rngs = jax.random.split(jax.random.PRNGKey(42), n_blocks)

    itercount = itertools.count()
    bijectors = []

    X_g = X.copy()

    pbar = tqdm.tqdm(block_rngs)
    with pbar:
        for iblock, irng in enumerate(pbar):

            pbar.set_description(
                f"Initializing - Block: {iblock+1} | Layer {next(itercount)}"
            )

            # ======================
            # MIXTURECDF
            # ======================
            # create keys for all inits
            irng, icdf_rng = jax.random.split(irng, 2)

            # intialize bijector and transformation
            X_g, layer = init_mixcdf_f.transform_and_bijector(
                inputs=X_g, rng=icdf_rng, n_features=n_features
            )

            # add bijector to list
            bijectors.append(layer)

            # ======================
            # LOGIT
            # ======================

            pbar.set_description(
                f"Initializing - Block: {iblock+1} | Layer {next(itercount)}"
            )

            # intialize bijector and transformation
            X_g, layer = init_icdf_f.transform_and_bijector(inputs=X_g)

            bijectors.append(layer)

            # ======================
            # HOUSEHOLDER
            # ======================
            pbar.set_description(
                f"Initializing - Block: {iblock+1} | Layer {next(itercount)}"
            )
            # create keys for all inits
            irng, hh_rng = jax.random.split(irng, 2)

            # intialize bijector and transformation
            X_g, layer = init_hh_f.transform_and_bijector(
                inputs=X_g, rng=hh_rng, n_features=n_features
            )

            bijectors.append(layer)

    # create base dist
    base_dist = Normal(jnp.zeros((n_features,)), jnp.ones((n_features,)))

    # create flow model
    gf_model = GaussianizationFlow(base_dist=base_dist, bijectors=bijectors)
    return gf_model


def init_gf_spline_model(
    shape: tuple,
    X: Array = None,
    n_blocks: int = 4,
    n_bins: int = 20,
    range_min: float = -10,
    range_max: float = 10,
    init_rotation: str = "pca",
    n_reflections: int = 10,
    plot_layers: bool = False,
    plot_blocks: bool = False,
    **kwargs,
):

    n_features = shape[0]
    rng = jax.random.PRNGKey(42)
    # rng, _ = jax.random.split(jax.random.PRNGKey(123), 2)

    # =====================
    # RQ Spline
    # ======================
    init_rq_f = InitPiecewiseRationalQuadraticCDF(
        n_bins=n_bins, range_min=range_min, range_max=range_max, **kwargs
    )
    # =====================
    # HouseHolder Transform
    # ======================
    n_reflections = n_reflections
    # initialize init function
    init_hh_f = InitHouseHolder(n_reflections=n_reflections, method=init_rotation)

    block_rngs = jax.random.split(rng, num=n_blocks)
    # rng = jax.random.split(jax.random.PRNGKey(42), n_blocks)
    # block_rngs = jax.random.split(jax.random.PRNGKey(42), n_blocks)

    itercount = itertools.count()
    bijectors = []

    X_g = X.copy()

    if plot_blocks:
        fig = corner.corner(np.array(X_g), color="red", hist_bin_factor=2)

    pbar = tqdm.tqdm(block_rngs)
    with pbar:
        for iblock, irng in enumerate(pbar):

            pbar.set_description(
                f"Initializing - Block: {iblock+1} | Layer {next(itercount)}"
            )

            # ======================
            # RQ Spline
            # ======================
            # create keys for all inits
            irng, irq_rng = jax.random.split(irng, 2)

            # intialize bijector and transformation
            X_g, layer = init_rq_f.transform_and_bijector(
                inputs=X_g, rng=irq_rng, shape=X.shape[1:]
            )

            # plot data
            if plot_layers and plot_blocks:
                fig = corner.corner(np.array(X_g), color="red", hist_bin_factor=2)

            # add bijector to list
            bijectors.append(layer)

            # ======================
            # HOUSEHOLDER
            # ======================
            pbar.set_description(
                f"Initializing - Block: {iblock+1} | Layer {next(itercount)}"
            )
            # create keys for all inits
            irng, hh_rng = jax.random.split(irng, 2)

            # intialize bijector and transformation
            X_g, layer = init_hh_f.transform_and_bijector(
                inputs=X_g, rng=hh_rng, n_features=n_features
            )

            bijectors.append(layer)

            # plot data
            if plot_blocks:
                fig = corner.corner(np.array(X_g), color="red", hist_bin_factor=2)

    # create base dist
    base_dist = Normal(jnp.zeros((n_features,)), jnp.ones((n_features,)))

    # create flow model
    gf_model = GaussianizationFlow(base_dist=base_dist, bijectors=bijectors)
    return gf_model


def init_gf_composite_spline_model(
    shape: tuple,
    X: Array = None,
    n_blocks: int = 4,
    n_bins: int = 20,
    range_min: float = 0.0,
    range_max: float = 1.0,
    init_rotation: str = "random",
    n_reflections: int = 10,
    squeeze: str = "sigmoid",
    **kwargs,
):

    n_features = shape[0]
    rng = jax.random.PRNGKey(42)
    # rng, _ = jax.random.split(jax.random.PRNGKey(123), 2)
    # =====================
    # Composite Transform
    # ======================
    init_nl_forward_f = InitGaussCDF()
    init_nl_inverse_f = InitInverseGaussCDF()

    # =====================
    # RQ Spline
    # ======================
    init_rq_f = InitPiecewiseRationalQuadraticCDF(
        n_bins=n_bins, range_min=range_min, range_max=range_max, **kwargs
    )
    # =====================
    # HouseHolder Transform
    # ======================
    n_reflections = n_reflections
    # initialize init function
    init_hh_f = InitHouseHolder(n_reflections=n_reflections, method=init_rotation)

    block_rngs = jax.random.split(rng, num=n_blocks)
    # rng = jax.random.split(jax.random.PRNGKey(42), n_blocks)
    # block_rngs = jax.random.split(jax.random.PRNGKey(42), n_blocks)

    itercount = itertools.count()
    bijectors = []

    X_g = X.copy()

    pbar = tqdm.tqdm(block_rngs)
    with pbar:
        for iblock, irng in enumerate(pbar):

            pbar.set_description(
                f"Initializing - Block: {iblock+1} | Layer {next(itercount)}"
            )
            # ======================
            # Forward Squeezing Transform
            # ======================
            # intialize bijector and transformation
            X_g, layer = init_nl_forward_f.transform_and_bijector(inputs=X_g,)
            # add bijector to list
            bijectors.append(layer)
            # ======================
            # RQ Spline
            # ======================
            # create keys for all inits
            irng, irq_rng = jax.random.split(irng, 2)

            # intialize bijector and transformation
            X_g, layer = init_rq_f.transform_and_bijector(
                inputs=X_g, rng=irq_rng, shape=X.shape[1:]
            )

            # add bijector to list
            bijectors.append(layer)

            # ======================
            # Inverse Squeezing Transform
            # ======================
            # intialize bijector and transformation
            X_g, layer = init_nl_inverse_f.transform_and_bijector(inputs=X_g,)
            # add bijector to list
            bijectors.append(layer)

            # ======================
            # HOUSEHOLDER
            # ======================
            pbar.set_description(
                f"Initializing - Block: {iblock+1} | Layer {next(itercount)}"
            )
            # create keys for all inits
            irng, hh_rng = jax.random.split(irng, 2)

            # intialize bijector and transformation
            X_g, layer = init_hh_f.transform_and_bijector(
                inputs=X_g, rng=hh_rng, n_features=n_features
            )

            bijectors.append(layer)

    # create base dist
    base_dist = Normal(jnp.zeros((n_features,)), jnp.ones((n_features,)))

    # create flow model
    gf_model = GaussianizationFlow(base_dist=base_dist, bijectors=bijectors)
    return gf_model


def add_gf_model_args(parser):
    # ====================
    # Model Args
    # ====================
    parser.add_argument(
        "--model",
        type=str,
        default="rqsplines",
        help="Standardize Input Training Data",
    )
    parser.add_argument(
        "--n_blocks", type=int, default=4, help="Standardize Input Training Data",
    )
    parser.add_argument(
        "--n_init_samples",
        type=int,
        default=1_000,
        help="Standardize Input Training Data",
    )
    # ====================
    # Spline Arguments
    # ====================
    parser.add_argument(
        "--n_bins", type=int, default=20, help="Standardize Input Training Data"
    )
    parser.add_argument(
        "--range_min", type=int, default=-12, help="Standardize Input Training Data",
    )
    parser.add_argument(
        "--range_max", type=int, default=12, help="Standardize Input Training Data",
    )
    parser.add_argument(
        "--boundary_slopes",
        type=str,
        default="unconstrained",
        help="Standardize Input Training Data",
    )
    parser.add_argument(
        "--identity_init",
        type=bool,
        default=False,
        help="Standardize Input Training Data",
    )
    # ====================
    # Mixture CDF Args
    # ====================
    parser.add_argument(
        "--n_components", type=int, default=20, help="Standardize Input Training Data"
    )
    parser.add_argument(
        "--mixture",
        type=str,
        default="gaussian",
        help="Standardize Input Training Data",
    )
    parser.add_argument(
        "--init_mixcdf", type=str, default="gmm", help="Standardize Input Training Data"
    )
    # ====================
    # Quantile Args
    # ====================

    parser.add_argument(
        "--inverse_cdf",
        type=str,
        default="gaussian",
        help="Standardize Input Training Data",
    )
    # ====================
    # Rotation Args
    # ====================
    parser.add_argument(
        "--n_reflections", type=int, default=2, help="Standardize Input Training Data"
    )
    parser.add_argument(
        "--init_rotation",
        type=str,
        default="random",
        help="Standardize Input Training Data",
    )
    return parser


def add_gf_model_spline_args(parser):
    # ====================
    # Model Args
    # ====================
    parser.add_argument(
        "--n_blocks", type=int, default=4, help="Standardize Input Training Data",
    )
    parser.add_argument(
        "--n_init_samples",
        type=int,
        default=1_000,
        help="Standardize Input Training Data",
    )
    # ====================
    # Spline Arguments
    # ====================
    parser.add_argument(
        "--n_bins", type=int, default=20, help="Standardize Input Training Data"
    )
    parser.add_argument(
        "--range_min", type=int, default=-12, help="Standardize Input Training Data",
    )
    parser.add_argument(
        "--range_max", type=int, default=12, help="Standardize Input Training Data",
    )
    parser.add_argument(
        "--boundary_slopes",
        type=str,
        default="unconstrained",
        help="Standardize Input Training Data",
    )
    parser.add_argument(
        "--identity_init",
        type=bool,
        default=False,
        help="Standardize Input Training Data",
    )
    # ====================
    # Rotation Args
    # ====================
    parser.add_argument(
        "--n_reflections", type=int, default=2, help="Standardize Input Training Data"
    )
    parser.add_argument(
        "--init_rotation",
        type=str,
        default="random",
        help="Standardize Input Training Data",
    )
    return parser
