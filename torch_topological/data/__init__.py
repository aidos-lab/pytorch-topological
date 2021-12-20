"""Module for various data operations and data set creation strategies."""

from .shapes import make_annulus
from .shapes import make_double_annulus

from .spheres import create_sphere_dataset

__all__ = [
    'create_sphere_dataset'
    'make_annulus',
    'make_double_annulus',
]
