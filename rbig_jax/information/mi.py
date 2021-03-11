from collections import namedtuple
from rbig_jax.transforms.rotation import InitPCARotation
from rbig_jax.transforms.histogram import InitUniHistUniformize
from rbig_jax.information.total_corr import rbig_total_correlation
from rbig_jax.information.entropy import rbig_multivariate_entropy
from typing import Callable, Optional

import jax
import jax.numpy as np

RBIGMutualInfo = namedtuple("RBIGMutualInfo", ["mi_X", "mi_Y", "mi_XY"])


RBIGEntropy = namedtuple("RBIGEntropy", ["H_X", "H_Y", "H_XY", "mi_XY"])


def rbig_mutual_info(
    X: np.ndarray,
    Y: np.ndarray,
    nbins: Optional[int] = None,
    precision: int = 100,
    support_extension: int = 10,
    alpha: int = 1e-5,
    base: int = 2,
    **kwargs
) -> RBIGMutualInfo:

    # ===================
    # DATASET X
    # ===================

    # Dataset 1
    X_g, info_loss_X = rbig_total_correlation(
        X=X,
        nbins=nbins,
        support_extension=support_extension,
        precision=precision,
        alpha=alpha,
        base=base,
        return_all=True,
        **kwargs
    )

    # ===================
    # DATASET Y
    # ===================

    Y_g, info_loss_Y = rbig_total_correlation(
        X=Y,
        nbins=nbins,
        support_extension=support_extension,
        precision=precision,
        alpha=alpha,
        base=base,
        return_all=True,
        **kwargs
    )
    # ===================
    # DATASET XY
    # ===================
    XY = np.hstack([X_g, Y_g])

    info_loss_XY = rbig_total_correlation(
        X=XY,
        nbins=nbins,
        support_extension=support_extension,
        precision=precision,
        alpha=alpha,
        base=base,
        return_all=False,
        **kwargs
    )
    return RBIGMutualInfo(
        mi_X=info_loss_X.sum() * np.log(base),
        mi_Y=info_loss_Y.sum() * np.log(base),
        mi_XY=info_loss_XY,
    )


def rbig_mutual_info_sum(
    X: np.ndarray,
    Y: np.ndarray,
    nbins: Optional[int] = None,
    precision: int = 100,
    support_extension: int = 10,
    alpha: int = 1e-5,
    base: int = 2,
    **kwargs
) -> RBIGEntropy:

    # ===================
    # DATASET X
    # ===================

    # Dataset 1
    H_X = rbig_multivariate_entropy(
        X=X,
        nbins=nbins,
        support_extension=support_extension,
        precision=precision,
        alpha=alpha,
        base=base,
        **kwargs
    )

    # ===================
    # DATASET Y
    # ===================

    H_Y = rbig_multivariate_entropy(
        X=Y,
        nbins=nbins,
        support_extension=support_extension,
        precision=precision,
        alpha=alpha,
        base=base,
        **kwargs
    )
    # ===================
    # DATASET XY
    # ===================
    XY = np.hstack([X, Y])

    H_XY = rbig_multivariate_entropy(
        X=XY,
        nbins=nbins,
        support_extension=support_extension,
        precision=precision,
        alpha=alpha,
        base=base,
        **kwargs
    )
    return RBIGEntropy(H_X=H_X, H_Y=H_Y, H_XY=H_XY, mi_XY=H_X + H_Y - H_XY)
