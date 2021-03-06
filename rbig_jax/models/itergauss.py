from typing import Callable, Optional

import jax
import jax.numpy as np
import objax
from chex import Array
from rbig_jax.transforms.block import InitRBIGBlock
from rbig_jax.utils import reverse_dataclass_params


class IterativeGaussianization:
    def __init__(
        self,
        uni_uniformize: Callable,
        rot_transform: Callable,
        n_features: int,
        eps: float = 1e-5,
        max_layers: int = 1_000,
    ):
        # create Gaussinization block
        fit_transform_f, forward_f, grad_f, inverse_f = InitRBIGBlock(
            uni_uniformize, rot_transform, eps
        )
        self.max_layers = max_layers
        self.n_features = n_features
        self.block_fit_transform = fit_transform_f
        self.block_transform = forward_f
        self.block_inverse_transform = inverse_f
        self.block_gradient_transform = grad_f
        # self.block_forward = jax.partial(
        #     rbig_block_forward, marginal_gauss_f=gaussianize_f
        # )
        # self.block_transform = rbig_block_transform
        # self.block_inverse = rbig_block_inverse
        # self.block_gradient = rbig_block_transform_gradient
        # self.max_layers = max_layers

        # # INFORMATION THEORY LOSS
        # tol_dims = get_tolerance_dimensions(n_samples)
        # self.uni_ent_est = uni_ent_est
        # self.loss_f = jax.partial(
        #     information_reduction, uni_entropy=self.uni_ent_est, tol_dims=tol_dims, p=p
        # )

        # # jit arguments (much faster!)
        # if jitted:
        #     self.block_forward = jax.jit(self.block_forward)
        #     self.block_transform = jax.jit(self.block_transform)
        #     self.block_inverse = jax.jit(self.block_inverse)
        #     self.block_gradient = jax.jit(self.block_gradient)
        #     self.loss_f = jax.jit(self.loss_f)

    def fit(self, X):

        _ = self.fit_transform(X)

        return self

    def fit_transform(self, X: Array) -> Array:
        def f_fit_transform(inputs, i):
            return self.block_fit_transform(inputs)

        X, layer_params = jax.lax.scan(f_fit_transform, X, None, self.max_layers)

        # self.n_features = X.shape[1]

        # # initialize parameter storage
        # params = []
        # losses = []
        # i_layer = 0

        # # loop through
        # while i_layer < self.max_layers:

        #     loss = jax.partial(self.loss_f, X=X)

        #     # fix info criteria
        #     X, block_params = self.block_forward(X)

        #     info_red = loss(Y=X)

        #     # append Parameters
        #     params.append(block_params)
        #     losses.append(info_red)

        #     i_layer += 1

        # self.n_layers = i_layer
        self.params = layer_params
        # self.info_loss = np.array(losses)
        return X

    def transform(self, X: Array) -> Array:
        def f_apply(inputs, params):

            outputs = self.block_transform(params, inputs)
            return outputs, 0

        X, _ = jax.lax.scan(f_apply, X, self.params, None)
        return X

    def inverse_transform(self, X: Array) -> Array:
        def f_invapply(inputs, params):

            outputs = self.block_inverse_transform(params, inputs)
            return outputs, 0

        # reverse the parameters
        params_reversed = reverse_dataclass_params(self.params)

        X, _ = jax.lax.scan(f_invapply, X, params_reversed, None)
        return X

    def log_det_jacobian(self, X: Array) -> Array:
        def fscan_gradient(inputs, params):
            return self.block_gradient_transform(params, inputs)

        # loop through params
        _, X_ldj_layers = jax.lax.scan(fscan_gradient, X, self.params, None)

        # summarize the layers (L, N, D) -> (N, D)
        X_ldj = np.sum(X_ldj_layers, axis=0)

        return X_ldj

    def score_samples(self, X: Array) -> Array:
        def fscan_gradient(inputs, params):
            return self.block_gradient_transform(params, inputs)

        # loop through params
        X, X_ldj_layers = jax.lax.scan(fscan_gradient, X, self.params, None)

        # summarize the layers (L, N, D) -> (N, D)
        X_ldj = np.sum(X_ldj_layers, axis=0)

        # calculate log probability
        latent_prob = jax.scipy.stats.norm.logpdf(X)

        # log probability
        log_prob = (latent_prob + X_ldj).sum(-1)

        return log_prob

    def score(self, X: Array) -> Array:
        return -self.score_samples(X).mean()

    def sample(self, n_samples: int, generator=objax.random.Generator()) -> Array:

        # generate independent Gaussian samples
        X_gauss = objax.random.normal((n_samples, self.n_features), generator=generator)

        # inverse transformation
        return self.inverse_transform(X_gauss)

    # def total_correlation(self, base: int = 2) -> np.ndarray:
    #     return np.sum(self.info_loss) * np.log(base)

    # def entropy(self, X: np.ndarray, base: int = 2) -> np.ndarray:
    #     return self.uni_ent_est(X).sum() * np.log(base) - self.total_correlation(base)
