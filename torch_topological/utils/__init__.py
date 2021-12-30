"""Utilities module."""

from .general import is_iterable

from .summary_statistics import total_persistence
from .summary_statistics import persistent_entropy
from .summary_statistics import polynomial_function

__all__ = [
    'is_iterable',
    'persistent_entropy',
    'polynomial_function',
    'total_persistence'
]
