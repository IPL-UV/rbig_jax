from typing import Optional, Tuple, Union

import jax.numpy as jnp
import tensor_annotations.jax as tjax
from chex import Array, dataclass
from einops import rearrange
from jax.random import PRNGKey
from tensor_annotations import axes
from distrax._src.bijectors.bijector import Bijector as distaxBijector
from rbig_jax.transforms.base import Bijector, TransformInfo

Batch = axes.Batch
Channels = axes.Channels
Height = axes.Height
Width = axes.Width
Features = axes.Features
ImageInputs = tjax.Array4[Batch, Channels, Height, Width]
ImageOutput = tjax.Array4[Batch, Channels, Height, Width]
MLOutputs = tjax.Array2[Batch, Features]
Outputs = Union[ImageOutput, MLOutputs]


# @dataclass(mappable_dataclass=False,)
class SqueezeLayer(distaxBijector):
    def __init__(self, H: int, W: int, C: int, Wn: int, Hn: int, fh: int, fw: int):
        self.H = H
        self.W = W
        self.C = C

        self.Hn = Hn
        self.Wn = Wn
        self.fh = fh
        self.fw = fw

    def forward_and_log_det(self, inputs: ImageInputs, **kwargs,) -> ImageOutput:

        # batch size is independent

        B, *_ = inputs.shape

        # forward rearrange
        outputs = rearrange(
            inputs,
            "B (Hn fh) (Wn fw) C -> B fh fw (C Hn Wn)",
            B=B,
            C=self.C,
            Hn=self.Hn,
            Wn=self.Wn,
            fh=self.fh,
            fw=self.fw,
        )

        # logdet empty
        log_det = jnp.zeros((B,))
        log_det = jnp.expand_dims(log_det, axis=1)
        return outputs, log_det

    def inverse_and_log_det(self, inputs: ImageInputs, **kwargs,) -> ImageInputs:

        B, *_ = inputs.shape

        # undo squeeze
        outputs = rearrange(
            inputs,
            "B fh fw (C Hn Wn) -> B (Hn fh) (Wn fw) C",
            C=self.C,
            Hn=self.Hn,
            Wn=self.Wn,
            fh=self.fh,
            fw=self.fw,
        )

        # logdet empty
        log_det = jnp.zeros((outputs.shape[0],))
        log_det = jnp.expand_dims(log_det, axis=1)
        return outputs, log_det


# # @dataclass(mappable_dataclass=False,)
# class SqueezeLayer(distaxBijector):
#     C: int
#     H: int
#     W: int
#     Hn: int
#     Wn: int
#     fh: int
#     fw: int

#     def forward_and_log_det(self, inputs: ImageInputs, **kwargs,) -> ImageOutput:

#         # batch size is independent

#         B, *_ = inputs.shape

#         # forward rearrange
#         outputs = rearrange(
#             inputs,
#             "B (Hn fh) (Wn fw) C -> B fh fw (C Hn Wn)",
#             B=B,
#             C=self.C,
#             Hn=self.Hn,
#             Wn=self.Wn,
#             fh=self.fh,
#             fw=self.fw,
#         )

#         # logdet empty
#         log_det = jnp.zeros((B,))
#         log_det = jnp.expand_dims(log_det, axis=1)
#         return outputs, log_det

#     def inverse_and_log_det(self, inputs: ImageInputs, **kwargs,) -> ImageInputs:

#         B, *_ = inputs.shape

#         # undo squeeze
#         outputs = rearrange(
#             inputs,
#             "B fh fw (C Hn Wn) -> B (Hn fh) (Wn fw) C",
#             C=self.C,
#             Hn=self.Hn,
#             Wn=self.Wn,
#             fh=self.fh,
#             fw=self.fw,
#         )

#         # logdet empty
#         log_det = jnp.zeros((outputs.shape[0],))
#         log_det = jnp.expand_dims(log_det, axis=1)
#         return outputs, log_det


def InitSqueezeLayer(filter_shape: Tuple[int, int],):
    def init_func(rng: PRNGKey, shape: Tuple, **kwargs):
        H, W, C = shape

        # extract coordinates
        fh, fw = filter_shape

        # # check for no remainders
        # assert H % fh == 0
        # assert W % fw == 0

        # new shapes
        Hn = H // fh
        Wn = W // fw

        return SqueezeLayer(H=H, W=W, C=C, fh=fh, fw=fw, Hn=Hn, Wn=Wn)

    return init_func


class CollapseLayer(distaxBijector):
    def __init__(self, collapse_shape: str, H: int, W: int, C: int):
        self.collapse_shape = collapse_shape
        self.H = H
        self.W = W
        self.C = C

    def forward_and_log_det(self, inputs: ImageInputs, **kwargs,) -> MLOutputs:

        # batch size is independent

        B, *_ = inputs.shape

        # collapse dimensions
        outputs = rearrange(
            inputs,
            "B H W C ->" + self.collapse_shape,
            B=B,
            C=self.C,
            H=self.H,
            W=self.W,
        )

        # logdet empty
        log_det = jnp.zeros((B,))
        log_det = jnp.expand_dims(log_det, axis=1)
        return outputs, log_det

    def inverse_and_log_det(self, inputs: ImageInputs, **kwargs,) -> ImageOutput:

        B, *_ = inputs.shape

        # un-collapse dimensions
        outputs = rearrange(
            inputs, self.collapse_shape + "-> B H W C", C=self.C, H=self.H, W=self.W,
        )

        # logdet empty
        log_det = jnp.zeros((outputs.shape[0],))
        log_det = jnp.expand_dims(log_det, axis=1)
        return outputs, log_det


# @dataclass(mappable_dataclass=False)
# class CollapseLayer(distaxBijector):
#     def __init__(self, collapse_shape: str, H: int, W: int, C: int)
#     collapse_shape: str
#     H: int
#     W: int
#     C: int

#     def forward_and_log_det(self, inputs: ImageInputs, **kwargs,) -> MLOutputs:

#         # batch size is independent

#         B, *_ = inputs.shape

#         # collapse dimensions
#         outputs = rearrange(
#             inputs,
#             "B H W C ->" + self.collapse_shape,
#             B=B,
#             C=self.C,
#             H=self.H,
#             W=self.W,
#         )

#         # logdet empty
#         log_det = jnp.zeros((B,))
#         log_det = jnp.expand_dims(log_det, axis=1)
#         return outputs, log_det

#     def inverse_and_log_det(self, inputs: ImageInputs, **kwargs,) -> ImageOutput:

#         B, *_ = inputs.shape

#         # un-collapse dimensions
#         outputs = rearrange(
#             inputs, self.collapse_shape + "-> B H W C", C=self.C, H=self.H, W=self.W,
#         )

#         # logdet empty
#         log_det = jnp.zeros((outputs.shape[0],))
#         log_det = jnp.expand_dims(log_det, axis=1)
#         return outputs, log_det


def InitCollapseLayer(collapse: Optional[str] = None):

    collapse_shape = _get_collapse_shape(collapse)

    def init_func(rng: PRNGKey, shape: Tuple, **kwargs):
        H, W, C = shape

        return CollapseLayer(collapse_shape=collapse_shape, H=H, W=W, C=C,)

    return init_func


# def Squeeze(
#     filter_shape: Tuple[int, int],
#     collapse: Optional[str] = None,
#     return_outputs: bool = False,
# ):
#     collapse_shape = _get_collapse_shape(collapse)

#     def init_func(rng: PRNGKey, shape: Tuple, inputs: Optional[Array] = None, **kwargs):
#         _, H, W, C = shape

#         # extract coordinates
#         fh, fw = filter_shape

#         # check for no remainders
#         assert H % fh == 0
#         assert W % fw == 0

#         # new shapes
#         Hn = H // fh
#         Wn = W // fw

#         def forward_and_log_det(
#             params: dataclass, inputs: ImageInputs, **kwargs,
#         ) -> Union[ImageInputs, MLOutputs]:

#             # batch size is independent

#             B, *_ = inputs.shape

#             # forward rearrange
#             outputs = rearrange(
#                 inputs,
#                 "B (Hn fh) (Wn fw) C -> B fh fw (C Hn Wn)",
#                 B=B,
#                 C=C,
#                 Hn=Hn,
#                 Wn=Wn,
#                 fh=fh,
#                 fw=fw,
#             )

#             if collapse_shape is not None:
#                 B_, H_, W_, C_ = outputs.shape

#                 # collapse dimensions
#                 outputs = rearrange(
#                     outputs, "B H W C ->" + collapse_shape, B=B_, C=C_, H=H_, W=W_,
#                 )

#             # logdet empty
#             log_det = jnp.zeros_like(outputs)

#             return outputs, log_det

#         def inverse_and_log_det(
#             params: dataclass, inputs: ImageInputs, **kwargs,
#         ) -> Union[ImageInputs, MLOutputs]:

#             if collapse_shape is not None:
#                 # un-collapse dimensions
#                 inputs = rearrange(
#                     inputs, collapse_shape + "-> B H W C", C=C * Hn * Wn, H=fh, W=fw,
#                 )

#             B, *_ = inputs.shape

#             # undo squeeze
#             outputs = rearrange(
#                 inputs,
#                 "B fh fw (C Hn Wn) -> B (Hn fh) (Wn fw) C",
#                 B=B,
#                 C=C,
#                 Hn=Hn,
#                 Wn=Wn,
#                 fh=fh,
#                 fw=fw,
#             )

#             # logdet empty
#             log_det = jnp.zeros_like(outputs)
#             return outputs, log_det

#         if return_outputs:
#             z, _ = forward_and_log_det((), inputs)
#             return z, (), forward_and_log_det, inverse_and_log_det
#         else:
#             return (), forward_and_log_det, inverse_and_log_det

#     return init_func


def _get_collapse_shape(collapse: Optional[str] = None) -> str:
    if collapse is "identity":
        collapse_shape = "B C H W"
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
