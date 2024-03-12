"""
    Contains the base interface for all models.
"""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .dataset import Dataset


class Model(ABC):
    """
    Base intreface for all models to be used for recommendation.
    It has 2 purposes:
        1. To specify how the model should fit the data
        2. To define the way the model performs a forward pass
    """

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Performs a forward pass with the provided data.
        """
        pass  # pragma: no cover

    @abstractmethod
    def fit(self, dataset: Dataset, *args, **kwargs):
        """
        Fit the model to the given dataset.

        Args:
            dataset (Dataset): The dataset to fit the model to.

        Returns:
            None
        """
        pass  # pragma: no cover


class SimilarityBased(Model):
    """
    A class that implements a similarity based model.
    """

    def __init__(
        self,
        similarity_measure: Callable = cosine_similarity,
    ):
        """
        Initialize the class with the given distance measure.

        Args:
            distance_measure (Callable, optional): The distance measure to use. Defaults to cosine_similarity.
        """
        self.similarity_measure = similarity_measure

    def forward(self, vector: np.array) -> np.array:
        """
        Calculates the distances between the input vector and the data,
        sorts the distances in descending order, and returns the indices.
        """
        vector = vector.reshape(1, -1)
        similarities = self.similarity_measure(vector, self.data)[0]
        indicies = np.argsort(similarities)[::-1]
        return indicies

    def fit(self, dataset: Dataset) -> None:
        """
        Fits the model to the provided dataset by translating the data into feature space.

        Parameters:
            dataset (Dataset): The dataset to fit the model to.

        Returns:
            None
        """
        self.data = dataset.features
