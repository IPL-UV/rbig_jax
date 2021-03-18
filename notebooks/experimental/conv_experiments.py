#%%

from pathlib import Path
import sys, os
from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".here"])

# append to path
sys.path.append(str(root))

# jax packages
import jax
import jax.numpy as jnp
from jax.config import config

# import chex
config.update("jax_enable_x64", False)

import numpy as np
from functools import partial
import objax
import chex
from pprint import pprint

# library functions
from rbig_jax.data import get_classic
from rbig_jax.plots import plot_joint, plot_joint_prob, plot_info_loss
from rbig_jax.transforms.parametric.mixture import MixtureGaussianCDF
from rbig_jax.transforms.logit import Logit
from rbig_jax.transforms.inversecdf import InverseGaussCDF
from rbig_jax.transforms.parametric import HouseHolder
from rbig_jax.transforms.base import CompositeTransform
from rbig_jax.models.gaussflow import GaussianizationFlow


# logging
import tqdm
import wandb

# plot methods
import matplotlib.pyplot as plt
import seaborn as sns

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)

#%%
"""Generate Toy Data
Source: https://www.openml.org/d/40927
"""

from sklearn import datasets

save_dir = str(Path(root).joinpath("datasets/cifar10"))

X_, y_ = datasets.fetch_openml("CIFAR_10", data_home=save_dir, return_X_y=True)

# %%

X = X_.copy()
y = y_.copy()


#%%
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots()
plt.imshow(X[0].reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8))
ax.set_yticks([])
ax.set_xticks([])
plt.tight_layout()
plt.show()


#%%
"""Subsample Images"""
n_subsamples = 10
X_subsample, y_subsample = X[:n_subsamples], y[:n_subsamples]

#%%
"""Reshape Images (REFACTORED)"""
from einops import rearrange


def tensor_2_image(tensor: np.ndarray) -> np.ndarray:

    image = rearrange(tensor, "b (c h w) -> b h w c", c=3, h=32, w=32)

    return image


def image_2_tensor(image: np.ndarray) -> np.ndarray:
    tensor = rearrange(image, "b h w c -> b (c h w)", c=3, h=32, w=32)
    return tensor


# tensor 2 image
X_image = tensor_2_image(X_subsample)
# image 2 tensor
X_tensor = image_2_tensor(X_image)

np.testing.assert_array_equal(X_tensor, X_subsample)


# %%
"""Do some standard preprocessing"""
from sklearn.preprocessing import StandardScaler


def normalize_image(image: np.array) -> np.array:
    # divide by 255
    image /= 255

    # standardize with mean/std
    tensor = image_2_tensor(image)

    scaler = StandardScaler().fit(tensor)

    tensor = scaler.transform(tensor)

    # tensor to image
    image = tensor_2_image(tensor)
    return image, scaler


# get images
X_images = tensor_2_image(X_subsample)
# normalize images
X_images_scaled, scaler = normalize_image(X_images)

#%%
"""Visualize"""


fig, ax = plt.subplots()
plt.imshow(X_images_scaled[0])
ax.set_yticks([])
ax.set_xticks([])
plt.tight_layout()
plt.show()

# show_cifar_image(X_images_scaled[0], y_subsample[0], scaler)

#%%
"""Initialize convolutional layer"""

# %%
"""Size/Reshape Magicks

Given an Image: B x C x H x W


Standard 1x1 Convolution
------------------------

 what we want
kernel_1x1 := 3 x 3 x 1 x 1

jax lingo
kernel := C_in x C_out x H x W
kernel := 'IOHW'

"""

from jax.lax import conv_general_dilated

# convert to jax array
X_image_jax = jnp.array(X_images_scaled, dtype=jnp.float32)

# define the kernel
kernel = jnp.ones(shape=(3, 3), dtype=jnp.float32)

# better orthogonal kernel

X_image_transform = conv_general_dilated(
    lhs=X_image_jax,  # input
    rhs=kernel[..., None, None],  # kernel
    window_strides=(1, 1),
    padding="SAME",
    lhs_dilation=(1, 1),
    rhs_dilation=(1, 1),
    dimension_numbers=("NHWC", "IOHW", "NHWC"),
)

fig, ax = plt.subplots()
plt.imshow(X_image_transform[0])
ax.set_yticks([])
ax.set_xticks([])
plt.tight_layout()
plt.show()

#%%
"""Invertible???"""

kernel_inv = kernel.T

X_image_approx = conv_general_dilated(
    lhs=X_image_transform,  # input
    rhs=kernel_inv[..., None, None],  # kernel
    window_strides=(1, 1),
    padding="SAME",
    lhs_dilation=(1, 1),
    rhs_dilation=(1, 1),
    dimension_numbers=("NHWC", "IOHW", "NHWC"),
)

import chex

chex.assert_tree_all_close(X_image_approx, X_image_jax)

# fig, ax = plt.subplots()
# plt.imshow(X_image_approx[0])
# ax.set_yticks([])
# ax.set_xticks([])
# plt.tight_layout()
# plt.show()

#%%
"""Ortogonal Convolution"""

from rbig_jax.transforms.parametric.householder import householder_transform


from jax.lax import conv_general_dilated

# convert to jax array
X_image_jax = jnp.array(X_image, dtype=jnp.float32)

# define the kernel

key = jax.random.PRNGKey(123)
kernel_ortho = jax.nn.initializers.orthogonal()(
    key=key, shape=(3, 3), dtype=jnp.float32
)

# better orthogonal kernel

X_image_transform_ortho = conv_general_dilated(
    lhs=X_image_jax,  # input
    rhs=kernel_ortho,  # kernel
    window_strides=(1, 1),
    padding="SAME",
    lhs_dilation=(1, 1),
    rhs_dilation=(1, 1),
    dimension_numbers=("NHWC", "IOHW", "NHWC"),
)


kernel_ortho_inv = kernel_ortho.transpose((1, 0, 2, 3))

X_image_approx = conv_general_dilated(
    lhs=X_image_transform_ortho,  # input
    rhs=kernel_ortho_inv,  # kernel
    window_strides=(1, 1),
    padding="SAME",
    lhs_dilation=(1, 1),
    rhs_dilation=(1, 1),
    dimension_numbers=("NHWC", "IOHW", "NHWC"),
)

import chex

chex.assert_tree_all_close(X_image_approx, X_image_jax)


# %%
"""Size/Reshape Magicks - HxW

Given an Image: B x C x H x W


Valero's Convolution
--------------------
dims : b x c x h x w
---
image:   1 x 1 x 6 x 6
valeros: (1 x 4) x 3 x 3 -> (1 x 4) x 9
gdn :    (1 x 3 x 3) x 4  -> (1 x 9) x 4
---
1) consider a square image
image:   1 x 1 x n x n
2) assume the same total input/output dimension
3) assume height/width is divisible by the spatial patch
4) assume spatial patch is even if height and width is even
---
image_ := (1 x 1 x sp_ x sp_) x h_ x w_

image_ := 1 x (1 x sp_ x sp_) x h_ x w_   # THIS ONE??? (EMAN)

image_ := (1 x sp_ x sp_) x (1 x h_) x (1 x w_)

image_ := 1 x (sp_ x sp_) x (1 x h_) x (1 x w_)

"""
n_channels = 1
height = 6
width = 6
print(f"{int(n_channels)}x{int(height)}x{int(width)}")
X_demo = jax.random.normal(key, (1, n_channels, height, width))

# EVEN CASE
spatial_patch = 10
new_channel = spatial_patch ** 2
new_height = height / spatial_patch
new_width = width / spatial_patch

assert float(height) % float(spatial_patch) == 0
assert n_channels * height * width == new_channel * new_height * new_width

print(f"{int(spatial_patch)}x{int(new_height)}x{int(new_width)}")
# X_reshaped = rearrange(X_demo, "b h w c -> ")

#%%
"""CHANNELS + H + W

Given an Image: B x C x H x W


Valero's Convolution
--------------------
dims : b x c x h x w
---
image:   1 x 3 x 6 x 6
valeros: (1 x 4) x 3 x 3 -> (1 x 4) x 9
gdn :    (1 x 3 x 3) x 4  -> (1 x 9) x 4
---
1) consider a square image
image:   1 x 1 x n x n
2) assume the same total input/output dimension
3) assume height/width is divisible by the spatial patch
4) assume spatial patch is even if height and width is even
---
image_ := (1 x c x sp_ x sp_) x h_ x w_

image_ := 1 x (c x sp_ x sp_) x h_ x w_   # THIS ONE??? (EMAN)

image_ := (1 x sp_ x sp_) x (c x h_) x (c x w_)

image_ := 1 x (sp_ x sp_) x (c x h_) x (c x w_)





"""
n_channels = 9
height = 6
width = 6
print(f"{int(n_channels)}x{int(height)}x{int(width)}")
X_demo = jax.random.normal(key, (1, n_channels, height, width))

# EVEN CASE
spatial_patch = 3
new_height = height / spatial_patch
new_width = width / spatial_patch

old_image_dims = n_channels * height * width
new_image_dims = n_channels * spatial_patch * spatial_patch * new_height * new_width

assert float(height) % float(spatial_patch) == 0
assert old_image_dims == new_image_dims

print(
    f"({int(n_channels)}x{int(spatial_patch)}x{int(spatial_patch)})x{int(new_height)}x{int(new_width)}"
)
print(
    f"({int(n_channels * spatial_patch * spatial_patch)})x{int(new_height)}x{int(new_width)}"
)
print(
    f"({int(n_channels)}x{int(new_height)}x{int(new_width)})x{int(spatial_patch)}x{int(spatial_patch)}"
)
print(
    f"({int(n_channels*new_height*new_width)})x{int(spatial_patch)}x{int(spatial_patch)}"
)


"""
---
Examples: 

Images
------
image   := c x h x w
        := 1 x 28 x 28   (MNIST)
        := 3 x 32 x 32 (CIFAR)

EO Spatial-Temporal
-------------------
image   := t x lat x lon
        := 3 x 3 x 3
        := 6 x 3 x 3
        := 9 x 3 x 3
        := 6 x 6 x 6

"""
# X_reshaped = rearrange(X_demo, "b h w c -> ")


# # %%
# """Size/Reshape Magicks

# Given an Image: B x C x H x W


# Valero's Convolution
# --------------------
# dims : b x c x h x w
# ---
# image:   1 x 3 x 6 x 6
# valeros: (1 x 4) x 3 x 3 -> (1 x 4) x 9
# gdn :    (1 x 3 x 3) x 4  -> (1 x 9) x 4
# ---
# 1) consider a square image
# image:   1 x 1 x n x n
# 2) assume the same total input/output dimension
# 3) assume height/width is divisible by the spatial patch
# 4) assume spatial patch is even if height and width is even
# ---
# image_ := (b x c x sp_ x sp_) x h_ x w_

# image_ := b x (c x sp_ x sp_) x h_ x w_   # THIS ONE??? (EMAN)

# image_ := (b x sp_ x sp_) x (c x h_) x (c x w_)

# image_ := b x (sp_ x sp_) x (c x h_) x (c x w_)


# """
# n_channels = 3
# height = 32
# width = 32
# print(f"{int(n_channels)}x{int(height)}x{int(width)}")
# X_demo = jax.random.normal(key, (1, n_channels, height, width))

# # EVEN CASE
# spatial_patch = 4
# new_channel = spatial_patch ** 2
# new_height = height / spatial_patch
# new_width = width / spatial_patch

# assert float(height) % float(spatial_patch) == 0
# assert n_channels * height * width == new_channel * new_height * new_width

# print(f"{int(spatial_patch)}x{int(new_height)}x{int(new_width)}")
# # X_reshaped = rearrange(X_demo, "b h w c -> ")

# #%%
# X_v_reshape = rearrange(X_image_jax, "b h w c -> ")

# # define the kernel
# kernel = jnp.ones(shape=(3, 3), dtype=jnp.float32)

# # better orthogonal kernel

# X_image_transform = conv_general_dilated(
#     lhs=X_image_jax,  # input
#     rhs=kernel[..., None, None],  # kernel
#     window_strides=(1, 1),
#     padding="SAME",
#     lhs_dilation=(1, 1),
#     rhs_dilation=(1, 1),
#     dimension_numbers=("NHWC", "IOHW", "NHWC"),
# )

# # fig, ax = plt.subplots()
# # plt.imshow(X_image_transform[0])
# # ax.set_yticks([])
# # ax.set_xticks([])
# # plt.tight_layout()
# # plt.show()

# %%
