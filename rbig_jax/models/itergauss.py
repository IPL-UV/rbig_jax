from typing import Callable, Optional

import jax
import jax.numpy as np
import objax
from chex import Array, dataclass
from rbig_jax.transforms.block import InitRBIGBlock, RBIGBlockParams
from rbig_jax.utils import get_minimum_zeroth_element, reverse_dataclass_params
from rbig_jax.information.total_corr import init_information_reduction_loss


@dataclass
class InfoLossState:
    max_layers: int
    ilayer: int
    info_loss: Array


class IterativeGaussianization:
    def __init__(
        self,
        uni_uniformize: Callable,
        rot_transform: Callable,
        n_features: int,
        n_samples: int = 10_000,
        zero_tolerance: int = 50,
        zero_tolerance_buffer: int = 10,
        eps: float = 1e-5,
        max_layers: int = 10_000,
        p: float = 0.1,
    ):
        # create Gaussinization block
        fit_transform_f, forward_f, grad_f, inverse_f = InitRBIGBlock(
            uni_uniformize, rot_transform, eps
        )
        self.info_loss_f = init_information_reduction_loss(
            n_samples=n_samples, base=2, p=0.1
        )
        self.max_layers = max_layers
        self.zero_tolerance = zero_tolerance
        self.zero_tolerance_buffer = zero_tolerance_buffer
        self.n_features = n_features
        self.block_fit_transform = jax.jit(fit_transform_f)
        self.block_transform = forward_f
        self.block_inverse_transform = inverse_f
        self.block_gradient_transform = grad_f
        self.info_loss_f = jax.jit(
            init_information_reduction_loss(n_samples=n_samples, base=2, p=p)
        )
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

        window = np.ones(self.zero_tolerance) / self.zero_tolerance

        def condition(state):

            # rolling average
            x_cumsum_window = np.convolve(np.abs(state.info_loss), window, "valid")
            n_zeros = int(np.sum(np.where(x_cumsum_window > 0.0, 0, 1)))
            return jax.lax.ne(n_zeros, 1) or state.ilayer > state.max_layers

        state = InfoLossState(
            max_layers=self.max_layers, ilayer=0, info_loss=np.ones(self.max_layers)
        )
        X_g = X
        params = []
        while condition(state):

            layer_loss = jax.partial(self.info_loss_f, X_before=X_g)

            # compute
            X_g, layer_params = self.block_fit_transform(X_g)

            # get information reduction
            layer_loss = layer_loss(X_after=X_g)

            # update layer loss
            info_losses = jax.ops.index_update(
                state.info_loss, state.ilayer, layer_loss
            )
            state = InfoLossState(
                max_layers=self.max_layers,
                ilayer=state.ilayer + 1,
                info_loss=info_losses,
            )
            params.append(layer_params)

        params = RBIGBlockParams(
            support=np.stack([iparam.support for iparam in params]),
            quantiles=np.stack([iparam.quantiles for iparam in params]),
            empirical_pdf=np.stack([iparam.empirical_pdf for iparam in params]),
            support_pdf=np.stack([iparam.support_pdf for iparam in params]),
            rotation=np.stack([iparam.rotation for iparam in params]),
        )
        # self.n_layers = i_layer
        self.params = params
        self.info_loss = info_losses[: state.ilayer]
        self.n_layers = state.ilayer

        return X_g

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

    def total_correlation(self, base: int = 2) -> np.ndarray:
        return np.sum(self.info_loss)  # * np.log(base)

    def entropy(self, X: np.ndarray, base: int = 2) -> np.ndarray:
        return self.uni_ent_est(X).sum() * np.log(base) - self.total_correlation(base)
