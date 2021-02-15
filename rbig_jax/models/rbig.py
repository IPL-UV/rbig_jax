from collections import namedtuple
from typing import Callable

import jax
import jax.numpy as np
import objax
import seaborn as sns
import tqdm
from jax.scipy import stats

from rbig_jax.information.total_corr import information_reduction
from rbig_jax.stopping import info_red_cond
from rbig_jax.transforms.block import (
    forward_gauss_block_transform, inverse_gauss_block_transform,
    inverse_gauss_block_transform_constrained)

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)

InfoLoss = namedtuple("InfoLoss", ["layer", "loss", "total_corr"])


class RBIGFlow:
    def __init__(
        self,
        rbig_block: Callable,
        tol_layers: int = 10,
        max_layers: int = 1_000,
        p: float = 0.25,
    ):

        self.block_fit = rbig_block
        self.block_forward = forward_gauss_block_transform
        self.block_inverse = inverse_gauss_block_transform
        self.info_loss = jax.partial(information_reduction, p=p)
        self.max_layers = max_layers
        self.tol_layers = tol_layers

    def fit_transform(self, X):

        self.n_features = X.shape[1]

        # initialize parameter storage
        params = []
        losses = []
        i_layer = 0

        # initialize condition state
        state = (0, losses, self.tol_layers, self.max_layers)
        while info_red_cond(state):

            # fix info criteria
            loss_f = jax.partial(self.info_loss, X=X)
            X, block_params = self.block_fit(X)

            loss = loss_f(Y=X)

            # append Parameters
            losses.append(loss)
            params.append(block_params)

            # update the state
            state = (i_layer, losses, self.tol_layers, self.max_layers)

            i_layer += 1
        self.n_layers = i_layer
        self.losses = np.array(losses)
        self.params = params
        return X

    def transform(self, X):
        for iparams in tqdm.tqdm(self.params):
            X = self.block_forward(X, iparams)
        return X

    def inverse_transform(self, X):
        for iparams in tqdm.tqdm(self.params[::-1]):
            X = self.block_inverse(X, iparams)
        return X

    def sample(self, n_samples: int):

        X_gauss = objax.random.normal((n_samples, self.n_features))
        return self.inverse_transform(X_gauss)

    def sample_constrained(self, n_samples: int):

        X = objax.random.normal((n_samples, self.n_features))
        for iparams in tqdm.tqdm(self.params[::-1]):
            X = inverse_gauss_block_transform_constrained(X, iparams)
        return X


class RBIGFlowJit(RBIGFlow):
    def __init__(
        self,
        rbig_block: Callable,
        tol_layers: int = 10,
        max_layers: int = 1_000,
        p: float = 0.25,
    ):

        self.block_fit = jax.jit(rbig_block)
        self.block_forward = jax.jit(forward_gauss_block_transform)
        self.block_inverse = jax.jit(inverse_gauss_block_transform)
        self.info_loss = jax.jit(jax.partial(information_reduction, p=p))
        self.max_layers = max_layers
        self.tol_layers = tol_layers
