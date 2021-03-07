from collections import namedtuple
from typing import Callable, Tuple, Union

import jax
import jax.numpy as np
from chex import Array, dataclass

BisectionState = namedtuple(
    "BisectionState",
    ["x", "current_x", "current_y", "lower_bound", "upper_bound", "diff", "iteration"],
)


def reverse_dataclass_params(params: dataclass) -> dataclass:
    return jax.tree_map(lambda x: x[::-1], params)


def safe_log(x: Array) -> Array:
    return np.log(np.clip(x, a_min=1e-22))


def marginal_transform(f: Callable):
    return jax.vmap(f)


def get_minimum_zeroth_element(x: Array, window_size: int = 10) -> int:

    # window for the convolution
    window = np.ones(window_size) / window_size

    # rolling average
    x_cumsum_window = np.convolve(np.abs(x), window, "valid")

    # get minimum zeroth element
    min_idx = int(np.min(np.argwhere(x_cumsum_window == 0.0)[0]))
    return min_idx


def get_domain_extension(
    data: np.ndarray, extension: Union[float, int],
) -> Tuple[float, float]:
    """Gets the extension for the support
    
    Parameters
    ----------
    data : np.ndarray
        the input data to get max and minimum

    extension : Union[float, int]
        the extension
    
    Returns
    -------
    lb : float
        the new extended lower bound for the data
    ub : float
        the new extended upper bound for the data
    """

    # case of int, convert to float
    if isinstance(extension, int):
        extension = float(extension / 100)

    # get the domain
    domain = np.abs(np.max(data) - np.min(data))

    # extend the domain
    domain_ext = extension * domain

    # get the extended domain
    lb = np.min(data) - domain_ext
    up = np.max(data) + domain_ext

    return lb, up


def interp_dim(x_new, x, y):
    return jax.vmap(np.interp, in_axes=(0, 0, 0))(x_new, x, y)


def searchsorted(bin_locations, inputs, eps=1e-6):
    # add noise to prevent zeros
    # bin_locations = bin_locations[..., -1] + eps
    bin_locations = bin_locations + eps

    # find bin locations (parallel bisection search)

    # sum dim
    print("Bins:", bin_locations.shape)
    print("Inputs:", inputs[..., None].shape)
    input_bins = np.sum(inputs[..., None] >= bin_locations, axis=-1)

    return input_bins


def bisection_body(f: Callable, state: BisectionState) -> BisectionState:

    # get all values greater than y
    greater_than = state.current_x > state.x

    # get all values that are less than...???
    less_than = 1.0 - greater_than

    # get new x
    new_y = 0.5 * greater_than * (
        state.current_y + state.lower_bound
    ) + 0.5 * less_than * (state.current_y + state.upper_bound)

    # get new bounds
    new_lb = greater_than * state.lower_bound + less_than * state.current_y
    new_ub = greater_than * state.current_y + less_than * state.upper_bound

    # get forward solution
    current_y = new_y
    current_x = f(current_y)
    # get difference
    diff = current_x - state.x

    # i = val.iteration + 1

    return BisectionState(
        x=state.x.squeeze(),
        current_x=current_x.squeeze(),
        current_y=current_y.squeeze(),
        diff=diff.squeeze(),
        lower_bound=new_lb.squeeze(),
        upper_bound=new_ub.squeeze(),
        iteration=state.iteration + 1,
    )

    # return (
    #     x.squeeze(),
    #     current_x.squeeze(),
    #     current_y.squeeze(),
    #     new_lb.squeeze(),
    #     new_ub.squeeze(),
    #     diff.squeeze(),
    #     i + 1,
    # )


def bisection_search(
    f: Callable,
    x: Array,
    lower: Array,
    upper: Array,
    atol: float = 1e-8,
    max_iters: int = 1_000,
) -> Array:

    # initialize solution
    y = np.zeros_like(x)

    # condition function
    def condition(state):
        # maximum iterations reached
        max_iters_reached = np.where(state.iteration > max_iters, True, False)
        # tolerance met
        tolerance_reached = np.allclose(state.diff, 0.0, atol=atol)
        return ~(max_iters_reached | tolerance_reached)

    # initialize state
    init_diff = np.ones_like(x) * 10.0

    state_init = BisectionState(
        x=x.squeeze(),
        current_x=f(y).squeeze(),
        current_y=y.squeeze(),
        lower_bound=lower.squeeze(),
        upper_bound=upper.squeeze(),
        diff=init_diff.squeeze(),
        iteration=0,
    )

    bisection_function = jax.partial(bisection_body, f)

    # do while loop
    final_state = jax.lax.while_loop(condition, bisection_function, state_init)

    # return the real value
    return np.atleast_1d(final_state.current_y)


bisection_search_vmap = jax.vmap(
    bisection_search, in_axes=(None, 0, None, None, None, None)
)


def make_interior_uniform_probability(X, eps=None):
    """Convert data to probability values in the open interval between 0 and 1.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data matrix.
    eps : float, optional
        Epsilon for clipping, defaults to ``np.info(X.dtype).eps``
    Returns
    -------
    X : array, shape (n_samples, n_features)
        Data matrix after possible modification.
    """
    # X = check_floating(X)
    if eps is None:
        eps = np.finfo(X.dtype).eps
    return np.minimum(np.maximum(X, eps), 1 - eps)
