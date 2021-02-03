import jax


def invgausscdf_forward_transform(X):

    return jax.scipy.stats.norm.ppf(X)


def invgausscdf_inverse_transform(X):
    return jax.scipy.stats.norm.cdf(X)


def get_params(X):

    # forward transformation
    X = jax.scipy.stats.norm.ppf(X)

    # Jacobian
    log_det_jacobian = jax.scipy.stats.norm.logpdf(X)

    def forward_function(X):
        # get the
        return jax.scipy.stats.norm.ppf(X)

    def inverse_function(X):
        return jax.scipy.stats.norm.cdf(X)

    return X, log_det_jacobian, forward_function, inverse_function

