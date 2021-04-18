import jax.numpy as jnp
from typing import Tuple, Optional, Union
from jax.random import PRNGKey
from chex import Array, dataclass
from tensor_annotations import axes
import tensor_annotations.jax as tjax
from einops import rearrange
from rbig_jax.transforms.base import TransformInfo

Batch = axes.Batch
Channels = axes.Channels
Height = axes.Height
Width = axes.Width
Features = axes.Features
ImageInputs = tjax.Array4[Batch, Channels, Height, Width]
ImageOutput = tjax.Array4[Batch, Channels, Height, Width]
MLOutputs = tjax.Array2[Batch, Features]
Outputs = Union[ImageOutput, MLOutputs]


def Squeeze(
    filter_shape: Tuple[int, int],
    collapse: Optional[str] = None,
    return_outputs: bool = False,
):
    collapse_shape = _get_collapse_shape(collapse)

    def init_func(rng: PRNGKey, shape: Tuple, inputs: Optional[Array] = None, **kwargs):
        _, H, W, C = shape

        # extract coordinates
        fh, fw = filter_shape

        # check for no remainders
        assert H % fh == 0
        assert W % fw == 0

        # new shapes
        Hn = H // fh
        Wn = W // fw

        def forward_and_log_det(
            params: dataclass, inputs: ImageInputs, **kwargs,
        ) -> Union[ImageInputs, MLOutputs]:

            # batch size is independent

            B, *_ = inputs.shape

            # forward rearrange
            outputs = rearrange(
                inputs,
                "B (Hn fh) (Wn fw) C -> B fh fw (C Hn Wn)",
                B=B,
                C=C,
                Hn=Hn,
                Wn=Wn,
                fh=fh,
                fw=fw,
            )

            if collapse_shape is not None:
                B_, H_, W_, C_ = outputs.shape

                # collapse dimensions
                outputs = rearrange(
                    outputs, "B H W C ->" + collapse_shape, B=B_, C=C_, H=H_, W=W_,
                )

            # logdet empty
            log_det = jnp.zeros_like(outputs)

            return outputs, log_det

        def inverse_and_log_det(
            params: dataclass, inputs: ImageInputs, **kwargs,
        ) -> Union[ImageInputs, MLOutputs]:

            if collapse_shape is not None:
                # un-collapse dimensions
                inputs = rearrange(
                    inputs, collapse_shape + "-> B H W C", C=C * Hn * Wn, H=fh, W=fw,
                )

            B, *_ = inputs.shape

            # undo squeeze
            outputs = rearrange(
                inputs,
                "B fh fw (C Hn Wn) -> B (Hn fh) (Wn fw) C",
                B=B,
                C=C,
                Hn=Hn,
                Wn=Wn,
                fh=fh,
                fw=fw,
            )

            # logdet empty
            log_det = jnp.zeros_like(outputs)
            return outputs, log_det

        if return_outputs:
            z, _ = forward_and_log_det((), inputs)
            return z, (), forward_and_log_det, inverse_and_log_det
        else:
            return (), forward_and_log_det, inverse_and_log_det

    return init_func


def _get_collapse_shape(collapse: Optional[str] = None):
    if collapse is None:
        collapse_shape = None
    elif collapse == "spatial":
        collapse_shape = "(B C) (H W)"
    elif collapse == "height":
        collapse_shape = "(B C W) H"
    elif collapse == "width":
        collapse_shape = "(B C H) W"
    elif collapse == "channels":
        collapse_shape = "(B H W) C"
    elif collapse == "features":
        collapse_shape = "B (C H W)"
    elif collapse == "all":
        collapse_shape = "(B C H W)"
    else:
        raise ValueError(f"Unrecognized collapse shape: {collapse}")

    return collapse_shape


def _get_new_shapes(
    height: int, width: int, channels: int, filter_shape: Tuple[int, int]
):

    # extract coordinates
    fh, fw = filter_shape

    # check for no remainders
    assert height % fh == 0
    assert width % fw == 0

    # new height and width
    height_n = height // fh
    width_n = width // fw

    # calculate new height, width, channels
    height = fh
    width = fw
    channels *= height_n * width_n

    return height, width, channels
