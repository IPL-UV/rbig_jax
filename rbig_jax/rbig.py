from typing import Callable, Optional
import jax
import jax.numpy as np
from collections import namedtuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from rbig_jax.information.total_corr import information_reduction
from rbig_jax.transforms.rbig import rbig_init, forward_transform, inverse_transform


# forward_transform = jax.jit(forward_transform)
# inverse_transform = jax.jit(inverse_transform)
# information_reduction = jax.jit(information_reduction)


TrainState = namedtuple(
    "TrainState",
    [
        "n_layers",  # number of layers
        "info_loss",  # information loss
        "params",
        "X",
        "Xldj",
    ],
)


def rbig_init_trainer(
    X: np.ndarray,
    rbig_params_init: Callable,
    stopping_criteria: str = "info",
    tol_layers: int = 10,
    min_layers: int = 10,
    max_layers: int = 1_000,
):

    if stopping_criteria == "info":

        # init loss with stopping criteria
        init_loss = np.pad(np.zeros((max_layers,)), (tol_layers, 0))
        init_loss = jax.ops.index_update(
            init_loss, np.arange(0, tol_layers, dtype=int), 1.0
        )
        train_state = TrainState(
            n_layers=0, info_loss=init_loss, params=[], X=X, Xldj=np.zeros(X.shape)
        )

        def condition_func(state):

            # get relevant layers (moving window)
            layers = state.info_loss[state.n_layers : tol_layers + state.n_layers]

            # sum layers
            info_sum = np.sum(np.abs(layers))

            # condition - there needs to be some loss of info
            info_crit = info_sum > 0.0
            return info_crit

    elif stopping_criteria == "max":
        # init loss with
        train_state = TrainState(
            n_layers=0,
            info_loss=np.zeros((max_layers,)),
            params=[],
            X=X,
            Xldj=np.zeros(X.shape),
        )

        def condition_func(state):
            # stopping criteria
            stop_crit = jax.lax.bitwise_not(state.n_layers >= max_layers)

            return stop_crit

    else:
        raise ValueError(f"Unrecognized stopping_criteria: {stopping_criteria}")

    def body(state):
        X, Xldj, params = rbig_params_init(state.X)

        # calculate the information loss
        it = information_reduction(state.X, X)

        state = TrainState(
            n_layers=state.n_layers + 1,
            info_loss=jax.ops.index_update(
                state.info_loss, tol_layers + state.n_layers, it
            ),
            params=state.params + [params],
            X=X,
            Xldj=state.Xldj + Xldj,
        )
        return state

    return train_state, condition_func, body


class RBIG(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        method: str = "histogram",
        support_ext: int = 10,
        precision: int = 1_000,
        alpha: int = 1e-5,
        tol_layers: int = 10,
        stopping_criteria: str = "info",
        max_layers: int = 1_000,
    ) -> None:
        self.method = method
        self.support_ext = support_ext
        self.precision = precision
        self.alpha = alpha
        self.tol_layers = tol_layers
        self.stopping_criteria = stopping_criteria
        self.max_layers = max_layers

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> None:
        self.n_dimensions = X.shape[-1]
        # init RBIG params getter
        rbig_params_init = rbig_init(
            method=self.method,
            support_ext=self.support_ext,
            precision=self.precision,
            alpha=self.alpha,
        )

        # # compile function (faster)
        # rbig_params_init = jax.jit(rbig_params_init)
        # _ = rbig_params_init(X[:10])

        # init RBIG state
        train_state, cond_func, body = rbig_init_trainer(
            X,
            rbig_params_init=rbig_params_init,
            stopping_criteria=self.stopping_criteria,
            tol_layers=self.tol_layers,
            max_layers=self.max_layers,
        )
        while cond_func(train_state):
            train_state = body(train_state)

        self.params = train_state.params
        self.n_layers = train_state.n_layers
        self.info_loss = train_state.info_loss[
            self.tol_layers : self.tol_layers + train_state.n_layers
        ]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        for iparam in self.params:
            X, _ = forward_transform(iparam, X)
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        for iparam in reversed(self.params):
            X = inverse_transform(iparam, X)
        return X

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_det = np.zeros(X.shape)
        for iparam in self.params:
            X, X_ldj = forward_transform(iparam, X)
            log_det += X_ldj

        latent_prob = jax.scipy.stats.norm.logpdf(X).sum(axis=1)
        log_det = log_det.sum(axis=1)
        return latent_prob + log_det

    def score(self, X: np.ndarray) -> float:
        return -self.predict_proba(X).mean()

    def sample(self, n_samples: int = 100, seed: Optional[int] = None) -> np.ndarray:
        rng = check_random_state(seed)
        X = np.array(rng.randn(n_samples, self.n_dimensions))
        return self.inverse_transform(X)

