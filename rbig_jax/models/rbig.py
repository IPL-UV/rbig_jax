from collections import namedtuple
from rbig_jax.stopping import info_red_cond
from typing import Callable, Optional

from rbig_jax.transforms.histogram import get_hist_params
from rbig_jax.transforms.uniformize import uniformize_transform
from rbig_jax.transforms.marginal import marginal_transform_params
from rbig_jax.transforms.gaussianize import gaussianize_forward
from rbig_jax.transforms.gaussianize import (
    gaussianize_marginal_transform,
    gaussianize_marginal_inverse,
)
from rbig_jax.transforms.rbig import (
    rbig_block_forward,
    rbig_block_inverse,
    rbig_block_transform,
    rbig_block_transform_gradient,
)
import jax
import jax.numpy as np
from jax.scipy import stats
import tqdm
import objax
from rbig_jax.information.total_corr import (
    get_tolerance_dimensions,
    information_reduction,
)
from rbig_jax.information.entropy import histogram_entropy
from rbig_jax.transforms.histogram import get_hist_params

import seaborn as sns

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)

InfoLoss = namedtuple("InfoLoss", ["layer", "loss", "total_corr"])


class RBIGFlow:
    def __init__(
        self, gaussianize_f: Callable, max_layers: int = 1_000, jitted: bool = True
    ):
        # create Gaussinization block
        self.block_forward = jax.partial(
            rbig_block_forward, marginal_gauss_f=gaussianize_f
        )
        self.block_transform = rbig_block_transform
        self.block_inverse = rbig_block_inverse
        self.max_layers = max_layers

        # jit arguments (much faster!)
        if jitted:
            self.block_forward = jax.jit(self.block_forward)
            self.block_transform = jax.jit(self.block_transform)
            self.block_inverse = jax.jit(self.block_inverse)

    def fit_transform(self, X):

        self.n_features = X.shape[1]

        # initialize parameter storage
        params = []
        i_layer = 0

        # loop through
        while i_layer < self.max_layers:

            # fix info criteria
            X, block_params = self.block_forward(X)

            # append Parameters
            params.append(block_params)

            i_layer += 1

        self.params = params
        return X  # , params

    def transform(self, X):

        for iparams in self.params:
            X = self.block_transform(X, iparams)
        return X

    def inverse_transform(self, X):
        for iparams in self.params[::-1]:
            X = self.block_inverse(X, iparams)
        return X

    def log_det_jacobian(self, X):
        return NotImplementedError()

    def score_samples(self, X):
        return NotImplementedError()

    def score(self, X):
        return NotImplementedError()

    def sample(self, n_samples: int):

        X_gauss = objax.random.normal((n_samples, self.n_features))
        return self.inverse_transform(X_gauss)


class IterativeGaussianization:
    def __init__(
        self,
        gaussianize_f: Optional[Callable] = None,
        uni_ent_est: Optional[Callable] = None,
        p: int = 0.25,
        n_samples: int = 10_000,
        max_layers: int = 1_000,
        jitted: bool = True,
    ):
        # create Gaussinization block
        self.block_forward = jax.partial(
            rbig_block_forward, marginal_gauss_f=gaussianize_f
        )
        self.block_transform = rbig_block_transform
        self.block_inverse = rbig_block_inverse
        self.block_gradient = rbig_block_transform_gradient
        self.max_layers = max_layers

        # INFORMATION THEORY LOSS
        tol_dims = get_tolerance_dimensions(n_samples)
        self.uni_ent_est = uni_ent_est
        self.loss_f = jax.partial(
            information_reduction, uni_entropy=self.uni_ent_est, tol_dims=tol_dims, p=p
        )

        # jit arguments (much faster!)
        if jitted:
            self.block_forward = jax.jit(self.block_forward)
            self.block_transform = jax.jit(self.block_transform)
            self.block_inverse = jax.jit(self.block_inverse)
            self.block_gradient = jax.jit(self.block_gradient)
            self.loss_f = jax.jit(self.loss_f)

    def fit(self, X):

        _ = self.fit_transform(X)

        return self

    def fit_transform(self, X):

        self.n_features = X.shape[1]

        # initialize parameter storage
        params = []
        losses = []
        i_layer = 0

        # loop through
        while i_layer < self.max_layers:

            loss = jax.partial(self.loss_f, X=X)

            # fix info criteria
            X, block_params = self.block_forward(X)

            info_red = loss(Y=X)

            # append Parameters
            params.append(block_params)
            losses.append(info_red)

            i_layer += 1

        self.n_layers = i_layer
        self.params = params
        self.info_loss = np.array(losses)
        return X

    def transform(self, X):

        for iparams in self.params:
            X = self.block_transform(X, iparams)
        return X

    def inverse_transform(self, X):
        for iparams in self.params[::-1]:
            X = self.block_inverse(X, iparams)
        return X

    def log_det_jacobian(self, X):
        X_ldj = np.zeros_like(X)

        # loop through params
        for iparams in self.params:
            X, log_det = self.block_gradient(X, iparams)
            X_ldj += log_det

        return X_ldj

    def score_samples(self, X):
        X_ldj = np.zeros_like(X)

        # loop through params
        for iparams in self.params:
            X, log_det = self.block_gradient(X, iparams)
            X_ldj += log_det

        # calculate log probability
        latent_prob = jax.scipy.stats.norm.logpdf(X)

        # log probability
        log_prob = (latent_prob + X_ldj).sum(-1)

        return log_prob

    def score(self, X):
        return self.score_samples(X).mean()

    def sample(self, n_samples: int):

        X_gauss = objax.random.normal((n_samples, self.n_features))
        return self.inverse_transform(X_gauss)

    def total_correlation(self, base: int = 2) -> np.ndarray:
        return np.sum(self.info_loss) * np.log(base)

    def entropy(self, X: np.ndarray, base: int = 2) -> np.ndarray:
        return self.uni_ent_est(X).sum() * np.log(base) - self.total_correlation(base)


class RBIGStandard(IterativeGaussianization):
    def __init__(
        self,
        p: int = 0.25,
        n_samples: int = 10_000,
        max_layers: int = 1_000,
        jitted: bool = True,
    ):
        gaussianize_f = get_default_mg(n_samples, True)
        uni_ent_est = get_default_entropy(n_samples)

        super().__init__(
            gaussianize_f=gaussianize_f,
            uni_ent_est=uni_ent_est,
            p=p,
            n_samples=n_samples,
            max_layers=max_layers,
            jitted=jitted,
        )


def get_default_mg(n_samples: int, return_params: bool = True):

    support_extension = 10
    alpha = 1e-5
    precision = 100
    nbins = int(np.sqrt(n_samples))
    return_params = return_params

    uniformize_transform = jax.partial(
        get_hist_params,
        nbins=nbins,
        support_extension=support_extension,
        precision=precision,
        alpha=alpha,
        return_params=return_params,
    )

    uni_transform_f = jax.partial(
        marginal_transform_params, function=uniformize_transform
    )

    gaussianize_f = jax.partial(gaussianize_forward, uni_transform_f=uni_transform_f)

    return gaussianize_f


def get_default_entropy(n_samples: int):
    nbins = int(np.sqrt(n_samples))
    entropy_f = jax.partial(histogram_entropy, nbins=nbins, base=2)

    return entropy_f


# class RBIGFlow_old:
#     def __init__(
#         self,
#         rbig_block: Callable,
#         tol_layers: int = 10,
#         max_layers: int = 1_000,
#         p: float = 0.25,
#     ):

#         self.block_fit = rbig_block
#         self.block_forward = forward_gauss_block_transform
#         self.block_inverse = inverse_gauss_block_transform
#         self.info_loss = jax.partial(information_reduction, p=p)
#         self.max_layers = max_layers
#         self.tol_layers = tol_layers

#     def fit_transform(self, X):

#         self.n_features = X.shape[1]

#         # initialize parameter storage
#         params = []
#         losses = []
#         i_layer = 0

#         # initialize condition state
#         state = (0, losses, self.tol_layers, self.max_layers)
#         while info_red_cond(state):

#             # fix info criteria
#             loss_f = jax.partial(self.info_loss, X=X)
#             X, block_params = self.block_fit(X)

#             loss = loss_f(Y=X)

#             # append Parameters
#             losses.append(loss)
#             params.append(block_params)

#             # update the state
#             state = (i_layer, losses, self.tol_layers, self.max_layers)

#             i_layer += 1
#         self.n_layers = i_layer
#         self.losses = np.array(losses)
#         self.params = params
#         return X

#     def transform(self, X):
#         for iparams in tqdm.tqdm(self.params):
#             X = self.block_forward(X, iparams)
#         return X

#     def inverse_transform(self, X):
#         for iparams in tqdm.tqdm(self.params[::-1]):
#             X = self.block_inverse(X, iparams)
#         return X

#     def sample(self, n_samples: int):

#         X_gauss = objax.random.normal((n_samples, self.n_features))
#         return self.inverse_transform(X_gauss)

#     def sample_constrained(self, n_samples: int):

#         X = objax.random.normal((n_samples, self.n_features))
#         for iparams in tqdm.tqdm(self.params[::-1]):
#             X = inverse_gauss_block_transform_constrained(X, iparams)
#         return X

