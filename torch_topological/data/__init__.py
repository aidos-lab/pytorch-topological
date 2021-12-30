"""Module for various data operations and data set creation strategies."""

from .shapes import sample_from_annulus
from .shapes import sample_from_disk
from .shapes import sample_from_double_annulus
from .shapes import sample_from_sphere
from .shapes import sample_from_torus
from .shapes import sample_from_unit_cube

__all__ = [
    'sample_from_annulus',
    'sample_from_disk',
    'sample_from_double_annulus',
    'sample_from_sphere',
    'sample_from_torus',
    'sample_from_unit_cube',
]
