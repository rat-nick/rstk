from typing import Callable

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Model:
    """
    Base class for all models to be used for recommendation.
    Their sole purpose is to perform a forward pass with given data.
    """

    def __init__(self, data):
        self.data = data

    def forward(self, *args, **kwargs):
        """
        Method that should be implemented by all models.
        Performs a forward pass with the provided data.
        """
        pass

    @staticmethod
    def build(*args, **kwargs) -> "Model":
        pass


class KNN(Model):
    """
    A class that implements the KNN model.
    """

    def __init__(
        self,
        features,
        distance_measure: Callable = cosine_similarity,
    ):
        super().__init__(features)
        self.distance_measure = distance_measure

    def forward(self, vector: np.array) -> np.array:
        distances = self.distance_measure(vector.reshape(1, -1), self.data)
        indicies = np.argsort(distances, axis=1)[0][::-1]
        return indicies
