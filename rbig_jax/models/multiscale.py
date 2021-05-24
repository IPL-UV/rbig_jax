from typing import Iterable, Tuple

import jax
import jax.numpy as jnp
import tqdm
from chex import Array
from distrax._src.distributions.normal import Normal

from rbig_jax.models import GaussianizationFlow
from rbig_jax.models.gaussflow import (init_default_gf_model,
                                       init_gf_spline_model)
from rbig_jax.transforms.base import BijectorChain
from rbig_jax.transforms.multiscale import MultiScaleBijector
from rbig_jax.transforms.reshape import init_scale_function


def init_multiscale_flow(
    X: Array,
    image_shape,
    filters: Iterable[Tuple[int, int]],
    seed: int = 42,
    n_blocks: int = 4,
    n_components: int = 20,
    mixture: str = "logistic",
    init_mixcdf: str = "gmm",
    init_rotation: str = "pca",
    inverse_cdf: str = "logistic",
    n_reflections: int = 10,
    return_transform: bool = False,
):
    rng = jax.random.PRNGKey(seed)
    X_g_subset_ = X.copy()

    bijectors = []

    with tqdm.tqdm(filters) as pbar:
        # Loop through the scales
        for i, i_filter in enumerate(pbar):

            pbar.set_description(
                f"Filter: {i_filter} - Layer: {i} - X: {X_g_subset_.shape}- Initializing MixCDF"
            )

            # split keys for sublayer params
            rng, ilayer_rngs = jax.random.split(rng, num=2)

            ms_reshape = init_scale_function(i_filter, image_shape, batch=False)

            X_g_subset_ = ms_reshape.forward(X_g_subset_)

            # initialize Gaussianization function

            X_g, gf_model = init_default_gf_model(
                shape=X_g_subset_.shape[1:],
                X=X_g_subset_,
                n_blocks=n_blocks,
                n_components=n_components,
                mixture=mixture,
                init_mixcdf=init_mixcdf,
                init_rotation=init_rotation,
                inverse_cdf=inverse_cdf,
                n_reflections=n_reflections,
                return_transform=True,
            )

            # create bijector chain
            bijector_chain = BijectorChain(bijectors=gf_model.bijectors)

            # initialize multiscale bijector
            rescale_params = init_scale_function(i_filter, image_shape, batch=False)
            ms_bijector = MultiScaleBijector(
                bijectors=bijector_chain,
                squeeze=rescale_params.forward,
                unsqueeze=rescale_params.inverse,
            )

            bijectors.append(ms_bijector)

            X_g_subset_ = ms_reshape.inverse(X_g)

    # create flow model
    base_dist = Normal(loc=jnp.zeros(X.shape[1]), scale=jnp.ones(X.shape[1]))
    gf_model = GaussianizationFlow(base_dist=base_dist, bijectors=bijectors)

    if return_transform:
        return X_g_subset_, gf_model
    else:
        return gf_model


def init_multiscale_flow_spline(
    X: Array,
    image_shape,
    filters: Iterable[Tuple[int, int]],
    seed: int = 42,
    n_blocks: int = 4,
    n_bins: int = 4,
    range_min: float = -12.0,
    range_max: float = 12.0,
    boundary_slopes: str = "unconstrained",
    identity_init: bool = False,
    n_reflections: int = 10,
    init_rotation: str = "random",
    return_transform: bool = False,
):
    rng = jax.random.PRNGKey(seed)
    X_g_subset_ = X.copy()

    bijectors = []

    with tqdm.tqdm(filters) as pbar:
        # Loop through the scales
        for i, i_filter in enumerate(pbar):

            pbar.set_description(
                f"Filter: {i_filter} - Layer: {i} - X: {X_g_subset_.shape}- Initializing MixCDF"
            )

            # split keys for sublayer params
            rng, ilayer_rngs = jax.random.split(rng, num=2)

            ms_reshape = init_scale_function(i_filter, image_shape, batch=False)

            X_g_subset_ = ms_reshape.forward(X_g_subset_)

            # initialize Gaussianization function

            X_g, gf_model = init_gf_spline_model(
                shape=X_g_subset_.shape[1:],
                X=X_g_subset_,
                n_blocks=n_blocks,
                n_bins=n_bins,
                range_min=range_min,
                range_max=range_max,
                boundary_slopes=boundary_slopes,
                identity_init=identity_init,
                n_reflections=n_reflections,
                init_rotation=init_rotation,
                return_transform=True,
            )

            # create bijector chain
            bijector_chain = BijectorChain(bijectors=gf_model.bijectors)

            # initialize multiscale bijector
            rescale_params = init_scale_function(i_filter, image_shape, batch=False)
            ms_bijector = MultiScaleBijector(
                bijectors=bijector_chain,
                squeeze=rescale_params.forward,
                unsqueeze=rescale_params.inverse,
            )

            bijectors.append(ms_bijector)

            X_g_subset_ = ms_reshape.inverse(X_g)

    # create flow model
    base_dist = Normal(loc=jnp.zeros(X.shape[1]), scale=jnp.ones(X.shape[1]))
    gf_model = GaussianizationFlow(base_dist=base_dist, bijectors=bijectors)

    if return_transform:
        return X_g_subset_, gf_model
    else:
        return gf_model
