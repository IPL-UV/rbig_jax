"""Code taken from:
https://github.com/jfcrenshaw/pzflow/blob/main/pzflow/bijectors/neural_splines.py
"""

from typing import Tuple
import jax.numpy as jnp
from jax.nn import softmax, softplus
from chex import dataclass, Array


@dataclass
class SplineParams:
    widths: Array
    heights: Array
    derivatives: Array


def rational_quadratic_spline(
    inputs: Array,
    W: Array,
    H: Array,
    D: Array,
    B: float,
    periodic: bool = False,
    inverse: bool = False,
) -> Tuple[Array, Array]:
    """Apply rational quadratic spline to inputs and return outputs with log_det.
    Applies the piecewise rational quadratic spline developed in [1].
    Parameters
    ----------
    inputs : Array
        The inputs to be transformed. shape=(n_samples, n_features)
    W : Array
        The widths of the spline bins. shape=(1, n_features, n_bins)
    H : Array
        The heights of the spline bins. shape=(1, n_features, n_bins)
    D : Array
        The derivatives of the inner spline knots. shape=(1, n_features, n_bins)
    B : float
        Range of the splines.
        Outside of (-B,B), the transformation is just the identity.
    inverse : bool, default=False
        If True, perform the inverse transformation.
        Otherwise perform the forward transformation.
    periodic : bool, default=False
        Whether to make this a periodic, Circular Spline [2].

    Returns
    -------
    outputs : Array
        The result of applying the splines to the inputs.
    log_det : Array
        The log determinant of the Jacobian at the inputs.

    References
    ----------
    [1] Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios.
        Neural Spline Flows. arXiv:1906.04032, 2019.
        https://arxiv.org/abs/1906.04032
    [2] Rezende, Danilo Jimenez et al.
        Normalizing Flows on Tori and Spheres. arxiv:2002.02428, 2020
        http://arxiv.org/abs/2002.02428
    """
    # knot x-positions
    xk = jnp.pad(
        -B + jnp.cumsum(W, axis=-1),
        [(0, 0)] * (len(W.shape) - 1) + [(1, 0)],
        mode="constant",
        constant_values=-B,
    )
    # knot y-positions
    yk = jnp.pad(
        -B + jnp.cumsum(H, axis=-1),
        [(0, 0)] * (len(H.shape) - 1) + [(1, 0)],
        mode="constant",
        constant_values=-B,
    )
    # knot derivatives
    if periodic:
        dk = jnp.pad(D, [(0, 0)] * (len(D.shape) - 1) + [(1, 0)], mode="wrap")
    else:
        dk = jnp.pad(
            D,
            [(0, 0)] * (len(D.shape) - 1) + [(1, 1)],
            mode="constant",
            constant_values=1,
        )
    # knot slopes
    sk = H / W

    # if not periodic, out-of-bounds inputs will have identity applied
    # if periodic, we map the input into the appropriate region inside
    # the period. For now, we will pretend all inputs are periodic.
    # This makes sure that out-of-bounds inputs don't cause problems
    # with the spline, but for the non-periodic case, we will replace
    # these with their original values at the end
    out_of_bounds = (inputs <= -B) | (inputs >= B)
    masked_inputs = jnp.where(out_of_bounds, jnp.abs(inputs) - B, inputs)

    # find bin for each input
    if inverse:
        idx = jnp.sum(yk <= masked_inputs[..., None], axis=-1)[..., None] - 1
    else:
        idx = jnp.sum(xk <= masked_inputs[..., None], axis=-1)[..., None] - 1

    # get kx, ky, kyp1, kd, kdp1, kw, ks for the bin corresponding to each input
    input_xk = jnp.take_along_axis(xk, idx, -1)[..., 0]
    input_yk = jnp.take_along_axis(yk, idx, -1)[..., 0]
    input_dk = jnp.take_along_axis(dk, idx, -1)[..., 0]
    input_dkp1 = jnp.take_along_axis(dk, idx + 1, -1)[..., 0]
    input_wk = jnp.take_along_axis(W, idx, -1)[..., 0]
    input_hk = jnp.take_along_axis(H, idx, -1)[..., 0]
    input_sk = jnp.take_along_axis(sk, idx, -1)[..., 0]

    if inverse:
        # [1] Appendix A.3
        # quadratic formula coefficients
        a = (input_hk) * (input_sk - input_dk) + (masked_inputs - input_yk) * (
            input_dkp1 + input_dk - 2 * input_sk
        )
        b = (input_hk) * input_dk - (masked_inputs - input_yk) * (
            input_dkp1 + input_dk - 2 * input_sk
        )
        c = -input_sk * (masked_inputs - input_yk)

        relx = 2 * c / (-b - jnp.sqrt(b ** 2 - 4 * a * c))
        outputs = relx * input_wk + input_xk
        # if not periodic, replace out-of-bounds values with original values
        if not periodic:
            outputs = jnp.where(out_of_bounds, inputs, outputs)

        # [1] Appendix A.2
        # calculate the log determinant
        dnum = (
            input_dkp1 * relx ** 2
            + 2 * input_sk * relx * (1 - relx)
            + input_dk * (1 - relx) ** 2
        )
        dden = input_sk + (input_dkp1 + input_dk - 2 * input_sk) * relx * (1 - relx)
        log_det = 2 * jnp.log(input_sk) + jnp.log(dnum) - 2 * jnp.log(dden)
        # if not periodic, replace log_det for out-of-bounds values = 0
        if not periodic:
            log_det = jnp.where(out_of_bounds, 0, log_det)
        log_det = log_det.sum(axis=1)

        return outputs, -log_det

    else:
        # [1] Appendix A.1
        # calculate spline
        relx = (masked_inputs - input_xk) / input_wk
        num = input_hk * (input_sk * relx ** 2 + input_dk * relx * (1 - relx))
        den = input_sk + (input_dkp1 + input_dk - 2 * input_sk) * relx * (1 - relx)
        outputs = input_yk + num / den
        # if not periodic, replace out-of-bounds values with original values
        if not periodic:
            outputs = jnp.where(out_of_bounds, inputs, outputs)

        # [1] Appendix A.2
        # calculate the log determinant
        dnum = (
            input_dkp1 * relx ** 2
            + 2 * input_sk * relx * (1 - relx)
            + input_dk * (1 - relx) ** 2
        )
        dden = input_sk + (input_dkp1 + input_dk - 2 * input_sk) * relx * (1 - relx)
        log_det = 2 * jnp.log(input_sk) + jnp.log(dnum) - 2 * jnp.log(dden)
        # if not periodic, replace log_det for out-of-bounds values = 0
        if not periodic:
            log_det = jnp.where(out_of_bounds, 0, log_det)
        log_det = log_det.sum(axis=1)

        return outputs, log_det
