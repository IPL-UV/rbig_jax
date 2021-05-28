from typing import Callable, NamedTuple, Optional, Tuple, Union

import jax.numpy as jnp
from chex import Array, dataclass
from rbig_jax.transforms.base import Bijector
from distrax._src.bijectors.bijector import Bijector as distaxBijector
from einops import rearrange
from jax.random import PRNGKey
from flax import struct


class RescaleParams(NamedTuple):
    fh: int
    fw: int
    Hn: int
    Wn: int
    W: int
    H: int
    C: int


class RescaleFunctions(NamedTuple):
    forward: Callable
    inverse: Callable
    params: RescaleParams


@struct.dataclass
class ReshapeTransform:
    bijector: Bijector
    squeeze: Callable = struct.field(pytree_node=False)
    unsqueeze: Callable = struct.field(pytree_node=False)

    def forward_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:

        # rescale data
        inputs = self.squeeze(inputs)

        # bijector chain transform
        outputs, logabsdet = self.bijector.forward_and_log_det(inputs)

        # unrescale data
        outputs = self.unsqueeze(outputs)
        logabsdet = self.unsqueeze(logabsdet)

        return outputs, logabsdet.sum(axis=1).reshape(-1, 1)

    def inverse_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:

        # rescale data
        inputs = self.squeeze(inputs)

        # bijector chain transform
        outputs, logabsdet = self.bijector.inverse_and_log_det(inputs)

        # unrescale data
        outputs = self.unsqueeze(outputs)
        logabsdet = self.unsqueeze(logabsdet)

        return outputs, logabsdet.sum(axis=1).reshape(-1, 1)

    def forward(self, inputs: Array) -> Array:

        # rescale data
        inputs = self.squeeze(inputs)
        # bijector chain transform
        outputs = self.bijector.forward(inputs)

        # unrescale data
        outputs = self.unsqueeze(outputs)

        return outputs

    def inverse(self, inputs: Array) -> Array:
        # rescale data
        inputs = self.squeeze(inputs)

        # bijector chain transform
        outputs = self.bijector.inverse(inputs)

        # unrescale data
        outputs = self.unsqueeze(outputs)
        return outputs

    def forward_log_det_jacobian(self, inputs: Array) -> Array:
        # rescale data
        inputs = self.squeeze(inputs)

        # bijector chain transform
        logabsdet = self.bijector.forward_log_det_jacobian(inputs)

        # unrescale data
        logabsdet = self.unsqueeze(logabsdet)
        return logabsdet.sum(axis=1).reshape(-1, 1)

    def inverse_log_det_jacobian(self, inputs: Array) -> Tuple[Array, Array]:

        # rescale data
        inputs = self.squeeze(inputs)

        # bijector chain transform
        logabsdet = self.bijector.inverse_log_det_jacobian(inputs)

        # unrescale data
        logabsdet = self.unsqueeze(logabsdet)

        return logabsdet.sum(axis=1).reshape(-1, 1)


def init_scale_function(filter, image_shape, batch: bool = True):

    # create filter params
    fh, fw = filter
    H = image_shape.H
    W = image_shape.W
    C = image_shape.C
    #     # do some checks!
    #     assert H / fh !% 0
    #     assert W / fw !% 0

    Hn = H // fh
    Wn = W // fw

    rescale_params = RescaleParams(fh=fh, fw=fw, H=H, W=W, C=C, Hn=Hn, Wn=Wn)

    if batch:

        def forward(inputs):

            temp = rearrange(
                inputs,
                "B (Hn fh Wn fw C) -> B Hn fh Wn fw C",
                C=C,
                Hn=Hn,
                Wn=Wn,
                fh=fh,
                fw=fw,
            )

            return rearrange(
                temp,
                "B Hn fh Wn fw C -> B Hn Wn (fh fw C)",
                fh=fh,
                fw=fw,
                C=C,
                Wn=Wn,
                Hn=Hn,
            )

        def inverse(inputs):

            temp = rearrange(
                inputs,
                "B Hn Wn (fh fw C) -> B Hn Wn fh fw C",
                #             B=inputs.shape[0],
                C=C,
                Hn=Hn,
                Wn=Wn,
                fh=fh,
                fw=fw,
            )
            return rearrange(
                temp,
                "B Hn Wn fh fw C -> B (Hn fh Wn fw C)",
                #             B=inputs.shape[0],
                C=C,
                Hn=Hn,
                Wn=Wn,
                fh=fh,
                fw=fw,
            )

    else:

        def forward(inputs):

            return rearrange(
                inputs,
                "B (Hn fh Wn fw C) -> (B Hn Wn) (fh fw C)",
                fh=fh,
                fw=fw,
                C=C,
                Wn=Wn,
                Hn=Hn,
            )

        def inverse(inputs):

            temp = rearrange(
                inputs,
                "(B Hn Wn) (fh fw C) -> B Hn Wn fh fw C",
                #             B=inputs.shape[0],
                C=C,
                Hn=Hn,
                Wn=Wn,
                fh=fh,
                fw=fw,
            )
            return rearrange(
                temp,
                "B Hn Wn fh fw C -> B (Hn fh Wn fw C)",
                #             B=inputs.shape[0],
                C=C,
                Hn=Hn,
                Wn=Wn,
                fh=fh,
                fw=fw,
            )

    return RescaleFunctions(forward=forward, inverse=inverse, params=rescale_params)


def _get_collapse_shape(collapse: Optional[str] = None) -> str:
    if collapse == "identity":
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


class ImageShape(NamedTuple):
    H: int
    W: int
    C: int


def flatten_image(img, shape: ImageShape, scaler=None, batch: bool = False) -> Array:

    # flatten image
    if batch:
        img = rearrange(img, "B H W C -> B (H W C)", C=shape.C, H=shape.H, W=shape.W)
    else:
        img = rearrange(img, "H W C -> (H W C)", C=shape.C, H=shape.H, W=shape.W)

    return img


def unflatten_image(img, shape: ImageShape, scaler=None, batch: bool = False) -> Array:
    # flatten image
    if batch:
        img = rearrange(img, "B (H W C) -> B H W C", C=shape.C, H=shape.H, W=shape.W)
    else:
        img = rearrange(img, "(H W C) -> H W C", C=shape.C, H=shape.H, W=shape.W)

    return img
