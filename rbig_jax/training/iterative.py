import time
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from chex import Array, dataclass
from distrax._src.distributions.normal import Normal

from rbig_jax.losses import IterativeInfoLoss, init_info_loss
from rbig_jax.models import GaussianizationFlow
from rbig_jax.transforms.block import RBIGBlockInit


def train_max_layers_model(
    X: Array,
    rbig_block: RBIGBlockInit,
    max_layers: int = 50,
    verbose: bool = False,
    interval: int = 10,
):
    """Simple training procedure using the iterative scheme.
    Uses a `max_layers` argument for the stopping criteria
    
    Parameters
    ----------
    X : Array
        the input data to be trained
    rbig_block : RBIGBlock
        a dataclass to be used
    max_layers : int, default=50
        the maximum number of layers to train the model
    verbose : bool
        whether to show
    interval : int
        how often to produce numbers
    
    Returns
    -------
    X_g : Array
        the transformed variable
    bijectors : List[Bijectors]
        a list of the bijectors which have been trained
    final_loss : Array
        an array for the loss function
    """
    # init rbig_block
    X_g = X.copy()
    n_features = X.shape[1]
    ilayer = 0

    n_bijectors = len(rbig_block.init_functions)
    bijectors = list()
    t0 = time.time()
    while ilayer < max_layers:

        # fit RBIG block
        X_g, ibijector = rbig_block.forward_and_bijector(X_g)

        # append bijectors
        bijectors += ibijector

        ilayer += 1

        if verbose:
            if ilayer % interval == 0:
                print(f"Layer {ilayer} - Elapsed Time: {time.time()-t0:.4f}")

    if verbose:
        print("Completed.")
        print(f"Final Number of layers: {ilayer} (Blocks: {ilayer//n_bijectors})")
        print(f"Elapsed Time: {time.time()-t0:.4f}")
    # ================================
    # Create Gaussianization model
    # ================================

    # create base distribution
    base_dist = Normal(jnp.zeros((n_features,)), jnp.ones((n_features,)))

    # create gaussianization flow model
    rbig_model = GaussianizationFlow(base_dist=base_dist, bijectors=bijectors)

    # return relevant stuff
    return X_g, rbig_model


def train_info_loss_model(
    X: Array,
    rbig_block_init: dataclass,
    max_layers: int = 1_000,
    zero_tolerance: int = 60,
    loss: Optional[IterativeInfoLoss] = None,
    verbose: bool = True,
    interval: int = 5,
    p: float = 0.25,
    n_layers_remove: Optional[int] = 10,
    jitted: bool = True,
) -> Tuple[List[dataclass], Array]:
    """Simple training procedure using the iterative scheme
    
    Parameters
    ----------
    X : Array
        the input data to be trained
    rbig_block_init : RBIGBlockInit
        a dataclass to be used
    loss : IterativeLoss
        a namedtuple with all of the components for the loss
    verbose : bool
        whether to show
    interval : int
        how often to produce numbers
    
    Returns
    -------
    X_g : Array
        the transformed variable
    bijectors : List[Bijectors]
        a list of the bijectors which have been trained
    final_loss : Array
        an array for the loss function
    """

    # init rbig_block
    X_g = X.copy()
    n_features = X.shape[1]

    # extract the components
    if loss is None:
        # initialize info loss function
        loss = init_info_loss(
            n_samples=X.shape[0],
            max_layers=max_layers,
            zero_tolerance=zero_tolerance,
            p=p,
            jitted=jitted,
        )
    loss_f, condition, state, name = loss

    assert name == "info"
    # assert state.zero_tolerance > remove_layers

    # initialize list of bijectors
    n_bijectors = len(rbig_block_init.init_functions)
    bijectors = list()
    t0 = time.time()
    while condition(state):

        # fit loss partially
        layer_loss = jax.partial(loss_f, X_before=X_g)

        # fit RBIG block
        X_g, ibijector = rbig_block_init.forward_and_bijector(X_g)

        # get information reduction
        layer_loss = layer_loss(X_after=X_g)

        # append bijectors
        bijectors += ibijector

        # update loss
        state = state.update_state(info_loss=layer_loss)

        if verbose:
            if state.ilayer % interval == 0:
                print(
                    f"Layer {state.ilayer} - Cum. Info Reduction: {state.info_loss[:state.ilayer].sum():.3f} - Elapsed Time: {time.time()-t0:.4f} secs"
                )

    if verbose:
        print(f"Converged at Layer: {state.ilayer}")
    # extract final loss
    final_loss = state.info_loss[: state.ilayer]

    # ================================
    # remove excess layers and loss
    # ================================
    if n_layers_remove is not None:
        final_loss, bijectors = _remove_layers(
            info_loss=final_loss,
            bijectors=bijectors,
            n_bijectors=n_bijectors,
            n_layers_remove=n_layers_remove,
        )

    t1 = time.time()

    # ================================
    # Create Gaussianization model
    # ================================

    # create base distribution
    base_dist = Normal(jnp.zeros((n_features,)), jnp.ones((n_features,)))

    # create gaussianization flow model
    rbig_model = GaussianizationFlow(base_dist=base_dist, bijectors=bijectors)

    rbig_model.info_loss = final_loss

    if verbose:
        print(
            f"Final Number of layers: {final_loss.shape[0]} (Blocks: {final_loss.shape[0]//n_bijectors})"
        )
        print(f"Total Time: {t1-t0:.4f} secs")

    # return relevant stuff
    return X_g, rbig_model


def _remove_layers(
    info_loss, bijectors, n_bijectors, n_layers_remove, buffer: int = 10
):

    if n_layers_remove + buffer < len(info_loss):

        # get total number of bijectors to remove
        n_bijectors_removed = n_bijectors * n_layers_remove

        # get new blocks
        bijectors = bijectors[:-n_bijectors_removed]

        # get new info los
        info_loss = info_loss[:-n_layers_remove]

    return info_loss, bijectors
