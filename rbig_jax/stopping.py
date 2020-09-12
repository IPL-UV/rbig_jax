import jax


def nll_loss(X, X_log_det):
    # calculate probability
    log_prob = jax.scipy.stats.norm.logpdf(X).sum(axis=1)

    likelihood = log_prob + X_log_det.sum(axis=1)
    # log likelihood
    return -likelihood.mean()
