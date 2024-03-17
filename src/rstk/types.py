"""
Module defining all custom data types that are used in the library.
"""

from typing import Any, Iterable, List, Tuple

import numpy as np
import pandas as pd


class Interaction:
    """
    Class representing a single interaction between a user and an item
    with a rating and an optional timestamp.
    """

    def __init__(
        self, user: int, item: int, rating: float, timestamp: int | None = None
    ):
        self.user = user
        self.item = item
        self.rating = rating
        self.timestamp = timestamp

    @property
    def tuple(self) -> Tuple[int, int, float, int | None]:
        return self.user, self.item, self.rating, self.timestamp

    @property
    def dict(self) -> dict[str, Any]:
        return {
            "user": self.user,
            "item": self.item,
            "rating": self.rating,
            "timestamp": self.timestamp,
        }

    @property
    def numpy(self) -> List[Any]:
        return np.array([self.user, self.item, self.rating, self.timestamp])


class InteractionMatrix:
    """
    Class for handling user-item interaction data.
    IDs are inner.
    """

    def __init__(
        self,
        matrix: np.ndarray = None,
        interactions: Iterable[Interaction] = None,
        dataframe: pd.DataFrame = None,
    ):
        if matrix is not None and len(matrix.shape) == 2 and matrix.shape[1] == 4:
            self.data = matrix
        elif interactions is not None:
            self.data = np.ndarray((len(interactions), 4))
            for i, interaction in enumerate(interactions):
                self.data[i] = interaction.numpy
        elif dataframe is not None:
            self.data = dataframe.to_numpy()
        else:
            raise ValueError("Interactions iterable or dataframe must be provided")

    @property
    def n_users(self) -> int:
        return len(np.unique(self.data[:, 0]))

    @property
    def n_items(self) -> int:
        return len(np.unique(self.data[:, 1]))

    @property
    def explicit(self) -> "ExplicitRatings":
        return ExplicitRatings(interaction_matrix=self)

    @property
    def implicit(self) -> "ImplicitRatings":
        return ImplicitRatings(interaction_matrix=self)


class Recommendations(List[Tuple[Any, float]]):
    """
    List of tuples where the first element is the item ID and the second element is the score.
    """

    def __init__(self):
        super().__init__()  # pragma: no cover


class ResponseDict(dict[str, Any]):
    """
    Dictionary representing a the response to be sent as a JSON
    """

    pass


class ExplicitRatings:
    """
    Class representing explicit ratings.
    """

    def __init__(self, interaction_matrix: InteractionMatrix = None):

        if interaction_matrix is not None:
            for interaction in interaction_matrix.data:
                self[int(interaction[1])] = (interaction[2], interaction[3])


class ImplicitRatings:
    """
    Class representing implicit ratings.
    """

    def __init__(
        self,
        interaction_matrix: InteractionMatrix = None,
        upper_threshold: float = 3.5,
        lower_threshold: float = 0,
    ):
        super().__init__()  # pragma: no cover

        if interaction_matrix is not None:
            for interaction in interaction_matrix.data:

                if interaction[2] >= upper_threshold:
                    value = 1
                elif interaction[2] < lower_threshold:
                    value = -1
                else:
                    value = 0

                self[interaction[1]] = (
                    value,
                    interaction[3],
                )
