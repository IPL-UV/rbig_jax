from sklearn.mixture import GaussianMixture
import numpy as np
import jax.numpy as jnp
import jax


def init_mixture_weights(rng, n_features, n_components, method, X=None):

    if method == "random":
        # initialize mixture
        prior_logits = jnp.ones((n_features, n_components)) / n_components
        means = jax.random.normal(key=rng, shape=(n_features, n_components))
        log_scales = jnp.zeros((n_features, n_components))

    elif method == "gmm":
        prior_logits, means, covariances = init_means_GMM_marginal(
            X, n_components=n_components
        )

        log_scales = jnp.array(covariances)
        prior_logits = jnp.array(prior_logits)
        means = jnp.array(means)

    else:
        raise ValueError(f"Unrecognized init method: {method}")
    return prior_logits, means, log_scales


def softplus_inverse(x):
    return jnp.log(jnp.exp(x) - 1.0)


def init_means_GMM_marginal(X: np.ndarray, n_components: int, **kwargs):
    """Initialize means with K-Means
    
    Parameters
    ----------
    X : np.ndarray
        (n_samples, n_features)
    n_components : int
        the number of clusters for the K-Means
    
    Returns
    -------
    clusters : np.ndarray
        (n_features, n_components)"""

    weights, means, covariances = [], [], []

    for iX in X.T:
        clf = GaussianMixture(
            n_components=n_components, covariance_type="diag", **kwargs
        ).fit(iX[:, None])
        weights.append(clf.weights_)
        means.append(clf.means_.T)
        covariances.append(clf.covariances_.T)

    return np.vstack(weights), np.vstack(means), np.vstack(covariances)
