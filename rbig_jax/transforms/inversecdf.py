import jax


def invgauss_forward_transform(X):

    return jax.scipy.stats.norm.ppf(X)


def invgauss_inverse_transform(X):
    return jax.scipy.stats.norm.cdf(X)
