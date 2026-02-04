"""Layers and loss terms for persistence-based optimisation."""

from .data import PersistenceInformation

from .distances import WassersteinDistance
from .sliced_wasserstein_distance import SlicedWassersteinDistance
from .sliced_wasserstein_kernel import SlicedWassersteinKernel
from .multi_scale_kernel import MultiScaleKernel

from .loss import SignatureLoss
from .loss import SummaryStatisticLoss

from .alpha_complex import AlphaComplex
from .cubical_complex import CubicalComplex
from .vietoris_rips_complex import VietorisRipsComplex

from .weighted_euler_characteristic_transform import WeightedEulerCurve
from .weighted_euler_characteristic_transform import EulerDistance

from .perslay import (PersLay,
                     PermutationEquivariant,
                     Image,
                     Landscape,
                     BettiCurve,
                     Entropy,
                     Exponential,
                     Rational,
                     RationalHat)

__all__ = [
    "AlphaComplex",
    "BettiCurve",
    "CubicalComplex",
    "Entropy",
    "EulerDistance",
    "Exponential",
    "Image",
    "Landscape",
    "MultiScaleKernel",
    "PersistenceInformation",
    "PermutationEquivariant",
    "PersLay",
    "Rational",
    "RationalHat",
    "SignatureLoss",
    "SlicedWassersteinDistance",
    "SlicedWassersteinKernel",
    "SummaryStatisticLoss",
    "VietorisRipsComplex",
    "WassersteinDistance",
    "WeightedEulerCurve"
]
