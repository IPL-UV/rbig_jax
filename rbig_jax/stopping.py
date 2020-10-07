from collections import namedtuple
import jax
import jax.numpy as np

# InfoLoss = namedtuple("InfoLoss", ["ilayer", "loss", "total_corr"])


def nll_loss(X, X_log_det):
    # calculate probability
    log_prob = jax.scipy.stats.norm.logpdf(X).sum(axis=1)

    likelihood = log_prob + X_log_det.sum(axis=1)
    # log likelihood
    return -likelihood.mean()


def info_red_cond(state):

    i_layers, losses, tol_layers, max_layers = state

    # condition 1 - max layers
    if i_layers >= max_layers:
        return False

    # condition 2 , less than tolerance layers
    if i_layers < tol_layers:
        return True

    # condition 3 - check losses
    info_loss = np.sum(np.abs(np.array(losses[-tol_layers:])))

    if info_loss == 0.0:
        return False
    else:
        return True

