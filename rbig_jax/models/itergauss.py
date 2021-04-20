# from typing import Callable, Optional

# import jax
# import jax.numpy as np
# import objax
# from chex import Array, dataclass

# from rbig_jax.information.total_corr import init_information_reduction_loss
# from rbig_jax.transforms.block import InitRBIGBlock, RBIGBlockParams
# from rbig_jax.transforms.histogram import (InitUniHistUniformize,
#                                            get_hist_params)
# from rbig_jax.transforms.rotation import InitPCARotation
# from rbig_jax.utils import get_minimum_zeroth_element, reverse_dataclass_params


# @dataclass
# class InfoLossState:
#     max_layers: int
#     ilayer: int
#     info_loss: Array


# class IterativeGaussianization:
#     def __init__(
#         self,
#         uni_uniformize: Callable,
#         rot_transform: Callable,
#         n_features: int,
#         n_samples: int = 10_000,
#         zero_tolerance: int = 50,
#         eps: float = 1e-5,
#         max_layers: int = 10_000,
#         p: float = 0.1,
#         base: int = 2,
#     ):
#         # create Gaussinization block
#         fit_transform_f, forward_f, grad_f, inverse_f = InitRBIGBlock(
#             uni_uniformize, rot_transform, eps
#         )
#         self.base = base
#         self.max_layers = max_layers
#         self.zero_tolerance = zero_tolerance
#         self.n_features = n_features
#         self.block_fit_transform = jax.jit(fit_transform_f)
#         self.block_transform = forward_f
#         self.block_inverse_transform = inverse_f
#         self.block_gradient_transform = grad_f
#         self.info_loss_f = jax.jit(
#             init_information_reduction_loss(n_samples=n_samples, base=self.base, p=p)
#         )

#     def fit(self, X):

#         _ = self.fit_transform(X)

#         return self

#     def fit_transform(self, X: Array) -> Array:

#         window = np.ones(self.zero_tolerance) / self.zero_tolerance

#         def condition(state):

#             # rolling average
#             x_cumsum_window = np.convolve(np.abs(state.info_loss), window, "valid")
#             n_zeros = int(np.sum(np.where(x_cumsum_window > 0.0, 0, 1)))
#             return jax.lax.ne(n_zeros, 1) or state.ilayer > state.max_layers

#         state = InfoLossState(
#             max_layers=self.max_layers, ilayer=0, info_loss=np.ones(self.max_layers)
#         )
#         X_g = X
#         params = []
#         while condition(state):

#             layer_loss = jax.partial(self.info_loss_f, X_before=X_g)

#             # compute
#             X_g, layer_params = self.block_fit_transform(X_g)

#             # get information reduction
#             layer_loss = layer_loss(X_after=X_g)

#             # update layer loss
#             info_losses = jax.ops.index_update(
#                 state.info_loss, state.ilayer, layer_loss
#             )
#             state = InfoLossState(
#                 max_layers=self.max_layers,
#                 ilayer=state.ilayer + 1,
#                 info_loss=info_losses,
#             )
#             params.append(layer_params)

#         params = RBIGBlockParams(
#             support=np.stack([iparam.support for iparam in params]),
#             quantiles=np.stack([iparam.quantiles for iparam in params]),
#             empirical_pdf=np.stack([iparam.empirical_pdf for iparam in params]),
#             support_pdf=np.stack([iparam.support_pdf for iparam in params]),
#             rotation=np.stack([iparam.rotation for iparam in params]),
#         )
#         # self.n_layers = i_layer
#         self.params = params
#         self.info_loss = info_losses[: state.ilayer]
#         self.n_layers = state.ilayer

#         return X_g

#     def transform(self, X: Array) -> Array:
#         def f_apply(inputs, params):

#             outputs = self.block_transform(params, inputs)
#             return outputs, 0

#         X, _ = jax.lax.scan(f_apply, X, self.params, None)
#         return X

#     def inverse_transform(self, X: Array) -> Array:
#         def f_invapply(inputs, params):

#             outputs = self.block_inverse_transform(params, inputs)
#             return outputs, 0

#         # reverse the parameters
#         params_reversed = reverse_dataclass_params(self.params)

#         X, _ = jax.lax.scan(f_invapply, X, params_reversed, None)
#         return X

#     def log_det_jacobian(self, X: Array) -> Array:
#         def fscan_gradient(inputs, params):
#             return self.block_gradient_transform(params, inputs)

#         # loop through params
#         _, X_ldj_layers = jax.lax.scan(fscan_gradient, X, self.params, None)

#         # summarize the layers (L, N, D) -> (N, D)
#         X_ldj = np.sum(X_ldj_layers, axis=0)

#         return X_ldj

#     def score_samples(self, X: Array) -> Array:
#         def fscan_gradient(inputs, params):
#             return self.block_gradient_transform(params, inputs)

#         # loop through params
#         X, X_ldj_layers = jax.lax.scan(fscan_gradient, X, self.params, None)

#         # summarize the layers (L, N, D) -> (N, D)
#         X_ldj = np.sum(X_ldj_layers, axis=0)

#         # calculate log probability
#         latent_prob = jax.scipy.stats.norm.logpdf(X)

#         # log probability
#         log_prob = (latent_prob + X_ldj).sum(-1)

#         return log_prob

#     def score(self, X: Array) -> Array:
#         return -self.score_samples(X).mean()

#     def sample(self, n_samples: int, generator=objax.random.Generator()) -> Array:

#         # generate independent Gaussian samples
#         X_gauss = objax.random.normal((n_samples, self.n_features), generator=generator)

#         # inverse transformation
#         return self.inverse_transform(X_gauss)

#     def total_correlation(self) -> np.ndarray:
#         return np.sum(self.info_loss) * np.log(self.base)

#     def entropy(self, X: np.ndarray, base: int = 2) -> np.ndarray:
#         raise NotImplementedError
#         # return self.uni_ent_est(X).sum() * np.log(base) - self.total_correlation(base)


# class RBIG(IterativeGaussianization):
#     def __init__(
#         self,
#         n_samples: int,
#         n_features: int,
#         support_extension: int = 10,
#         zero_tolerance: int = 30,
#         precision: int = 100,
#         alpha: int = 1e-5,
#         nbins: Optional[int] = None,
#         p: int = 0.25,
#         max_layers: int = 1_000,
#         eps: float = 1e-5,
#     ):
#         if nbins is None:
#             nbins = int(np.sqrt(n_samples))

#         # initialize histogram transformation
#         uni_uniformize = InitUniHistUniformize(
#             n_samples=n_samples,
#             nbins=nbins,
#             support_extension=support_extension,
#             precision=precision,
#             alpha=alpha,
#         )
#         # initialize rotation transformation
#         rot_transform = InitPCARotation()

#         super().__init__(
#             uni_uniformize=uni_uniformize,
#             rot_transform=rot_transform,
#             p=p,
#             n_samples=n_samples,
#             n_features=n_features,
#             max_layers=max_layers,
#             zero_tolerance=zero_tolerance,
#             eps=eps,
#         )


# def get_default_mg(n_samples: int, return_params: bool = True):

#     support_extension = 10
#     alpha = 1e-5
#     precision = 100
#     nbins = int(np.sqrt(n_samples))

#     uniformize_transform = jax.partial(
#         get_hist_params,
#         nbins=nbins,
#         support_extension=support_extension,
#         precision=precision,
#         alpha=alpha,
#         return_params=return_params,
#     )

#     uni_transform_f = jax.partial(
#         marginal_transform_params, function=uniformize_transform
#     )

#     gaussianize_f = jax.partial(gaussianize_forward, uni_transform_f=uni_transform_f)

#     return gaussianize_f


# def get_default_entropy(n_samples: int):
#     nbins = int(np.sqrt(n_samples))
#     entropy_f = jax.partial(histogram_entropy, nbins=nbins, base=2)

#     return entropy_f
