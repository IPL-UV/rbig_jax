import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from rbig_jax.transforms.kde import scotts_method


def init_mixture_weights(rng, n_features, n_components, method, X=None, **kwargs):

    if method == "random":
        # initialize mixture
        prior_logits = jnp.zeros((n_features, n_components))
        means = jax.random.normal(
            key=jax.random.PRNGKey(rng) if isinstance(rng, int) else rng,
            shape=(n_features, n_components),
        )
        log_scales = jnp.zeros((n_features, n_components))

    elif method == "gmm":
        prior_logits, means, covariances = init_means_GMM_marginal(
            X,
            n_components=n_components,
            random_state=rng if isinstance(rng, int) else int(rng[0]),
            covariance_type="diag",
            **kwargs,
        )
        log_scales = tfp.math.softplus_inverse(jnp.sqrt(covariances))

        prior_logits = jnp.array(prior_logits)
        prior_logits = jnp.log(prior_logits)

        means = jnp.array(means)

    elif method == "kmeans":

        # initialize means
        clusters = init_means_kmeans_marginal(
            X=X,
            n_components=n_components,
            random_state=rng if isinstance(rng, int) else int(rng[0]),
            **kwargs,
        )
        means = jnp.array(clusters)

        # initialize mixture distribution (uniform)
        prior_logits = jnp.ones((n_features, n_components))

        # initialize bandwith (rule of thumb estimator)
        # bandwith = scotts_method(n_samples=X.shape[0], n_features=n_features)
        log_scales = tfp.math.softplus_inverse(jnp.std(X, axis=0))
        log_scales = log_scales @ jnp.ones((n_features, n_components))

    else:
        raise ValueError(f"Unrecognized init method: {method}")
    return prior_logits, means, log_scales


def softplus_inverse(x):
    return jnp.log(jnp.exp(x) - 1.0)


def init_means_kmeans_marginal(X: np.ndarray, n_components: int, **kwargs):
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

    clusters = []

    for iX in X.T:
        clusters.append(
            KMeans(n_clusters=n_components, **kwargs)
            .fit(iX[:, None])
            .cluster_centers_.T
        )

    return np.vstack(clusters)


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
        clf = GaussianMixture(n_components=n_components, **kwargs).fit(iX[:, None])
        weights.append(clf.weights_)
        means.append(clf.means_.T)
        covariances.append(clf.covariances_.T)

    return np.vstack(weights), np.vstack(means), np.vstack(covariances)
