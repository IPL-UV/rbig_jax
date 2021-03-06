from typing import Callable
from chex import dataclass, Array
from rbig_jax.transforms.inversecdf import InitInverseGaussCDF
from rbig_jax.transforms.marginal import (
    marginal_fit_transform,
    marginal_gradient_transform,
    marginal_transform,
)


@dataclass
class RBIGBlockParams:
    support: Array
    quantiles: Array
    support_pdf: Array
    empirical_pdf: Array
    rotation: Array


def InitRBIGBlock(uni_uniformize: Callable, rot_transform: Callable, eps: float = 1e-5):

    # unpack functions
    uni_init_f, uni_forward_f, uni_grad_f, uni_inverse_f = uni_uniformize
    icdf_init_f, icdf_forward_f, icdf_grad_f, icdf_inverse_f = InitInverseGaussCDF(
        eps=eps
    )
    rot_init_f, rot_forward_f, _, rot_inverse_f = rot_transform

    # TODO a bin initialization function
    def init_func(inputs):

        # marginal uniformization
        inputs, uni_params = marginal_fit_transform(inputs, uni_init_f)

        # inverse CDF Transformation
        inputs, _ = icdf_init_f(inputs)

        # rotation
        outputs, rot_params = rot_init_f(inputs)

        # initialize new RBiG params
        params = RBIGBlockParams(
            support=uni_params.support,
            quantiles=uni_params.quantiles,
            empirical_pdf=uni_params.empirical_pdf,
            support_pdf=uni_params.support_pdf,
            rotation=rot_params.rotation,
        )

        return outputs, params

    def transform(params, inputs):

        # marginal uniformization
        inputs = marginal_transform(inputs, params, uni_forward_f)

        # inverse CDF transformation
        inputs = icdf_forward_f(params, inputs)

        # rotation
        outputs = rot_forward_f(params, inputs)

        return outputs

    def gradient_transform(params, inputs):

        # marginal uniformization
        inputs, Xu_ldj = marginal_gradient_transform(inputs, params, uni_grad_f)

        # inverse CDF transformation
        inputs, Xg_ldj = icdf_grad_f(params, inputs)

        # rotation is zero...
        outputs = rot_forward_f(params, inputs)

        # sum the log det jacobians
        logabsdet = Xu_ldj + Xg_ldj

        return outputs, logabsdet

    def inverse_transform(params, inputs):

        # rotation
        inputs = rot_inverse_f(params, inputs)

        # inverse gaussian cdf
        inputs = icdf_inverse_f(params, inputs)

        # marginal uniformization
        outputs = marginal_transform(inputs, params, uni_inverse_f)

        return outputs

    return init_func, transform, gradient_transform, inverse_transform
