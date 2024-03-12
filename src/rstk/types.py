"""
Module defining all custom data types that are used in the library.
"""

from typing import Any

import numpy as np


class Rating:
    """
    Class representing a rating of an item and an optional timestamp
    """

    def __init__(self, rating: float, timestamp: int | None = None):  # pragma: no cover
        self.rating = rating
        self.timestamp = timestamp


class ItemRatings(dict[Any, Rating]):
    """
    Dictionary of item ratings where the key is the raw item ID,
    and the value is a Rating object
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # pragma: no cover


class ExplicitRatings(ItemRatings):
    """
    Dictionary of item ratings where the key is the raw item ID,
    and the value is a Rating object
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # pragma: no cover


class ImplicitRatings(ItemRatings):
    """
    Class containing ratings of items where the key is the raw item ID,
    and the value is a float rating
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # pragma: no cover


class FeatureVector(np.ndarray):
    """
    Vector in feature space representing one item
    """


class ResponseDict(dict[str, Any]):
    """
    Dictionary representing a the response to be sent as a JSON
    """

    pass
