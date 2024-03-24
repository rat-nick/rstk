"""
Module defining the common data types used in the library.
Also contains the ConversionRegistry class for handling data conversions.
"""

from typing import Any, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .dataset import FeatureDataset


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

    def tuple(self) -> Tuple[int, int, float, int | None]:
        """
        Returns a tuple of the user, item, rating, and timestamp.

        Returns:
            Tuple[int, int, float, int | None]: The object viewed as a tuple.
        """
        return self.user, self.item, self.rating, self.timestamp

    def dict(self) -> dict[str, Any]:
        """
        Returns a dictionary of the user, item, rating, and timestamp.

        Returns:
            dict[str, Any]: The object viewed as a dictionary.
        """
        return {
            "user": self.user,
            "item": self.item,
            "rating": self.rating,
            "timestamp": self.timestamp,
        }

    def numpy(self) -> np.ndarray:
        """
        Returns a numpy array of the user, item, rating, and timestamp.

        Returns:
            np.ndarray: The object viewed as a numpy array.
        """
        return np.array([self.user, self.item, self.rating, self.timestamp])


class InteractionMatrix:
    """
    Class for handling user-item interaction data.
    IDs are inner. Should be used as the input type for Engine objects.
    """

    def __init__(
        self,
        shape: Tuple = None,
        matrix: np.ndarray = None,
        interactions: Iterable[Interaction] = None,
        dataframe: pd.DataFrame = None,
    ):
        # if more than one named parameter is provided
        num_args = 0
        num_args = num_args + 1 if matrix is not None else num_args
        num_args = num_args + 1 if interactions is not None else num_args
        num_args = num_args + 1 if dataframe is not None else num_args
        num_args = num_args + 1 if shape is not None else num_args

        if num_args != 1:
            raise ValueError(
                "Exactly one of shape, dataframe, matrix, or interactions must be provided"
            )
        if shape is not None:
            self.data = np.zeros(shape)
        elif matrix is not None and len(matrix.shape) == 2 and matrix.shape[1] == 4:
            self.data = matrix
        elif interactions is not None:
            self.data = np.ndarray((len(interactions), 4))
            for i, interaction in enumerate(interactions):
                self.data[i] = interaction.numpy()
        elif dataframe is not None:
            self.data = dataframe.to_numpy()
        else:
            raise ValueError(
                "shape, dataframe, matrix, or interactions must be provided"
            )

    @property
    def n_users(self) -> int:
        """
        Returns the number of unique users in the data.

        Returns:
            int: The number of unique users in the data.
        """
        return len(np.unique(self.data[:, 0]))

    @property
    def n_items(self) -> int:
        """
        Returns the number of unique items in the data.

        Returns:
            int: The number of unique items in the data.
        """
        return len(np.unique(self.data[:, 1]))

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def from_exp(explicit: "Explicit", *args, **kwargs) -> "InteractionMatrix":
        """
        Static method for creating an InteractionMatrix from an Explicit.

        Args:
            explicit (Explicit): The Explicit object.

        Returns:
            InteractionMatrix: The resulting InteractionMatrix.
        """
        im = InteractionMatrix((len(explicit), 4))
        for i, (key, value) in enumerate(explicit.items()):
            im.data[i] = np.asarray([0, key, value[0], value[1]])
        return im

    @staticmethod
    def from_seq(sequential: "Sequential") -> "InteractionMatrix":
        """
        Returns an InteractionMatrix from a Sequential object.

        Args:
            sequential (Sequential): The Sequential object.

        Returns:
            InteractionMatrix: The resulting InteractionMatrix.
        """
        pass


class Explicit(dict[int, Tuple[float, ...]]):
    """
    Class representing explicit ratings of a single user.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_im(matrix: InteractionMatrix, *args, **kwargs) -> "Explicit":
        """
        Static method for creating an Explicit from an InteractionMatrix.
        The InteractionMatrix provides item IDs and ratings.


        Args:
            matrix (InteractionMatrix): The InteractionMatrix.

        Returns:
            Explicit: The resulting Explicit.
        """
        res = {}
        for interaction in matrix:
            res[int(interaction[1])] = (interaction[2], interaction[3])
        return Explicit(res)


class Implicit(dict[int, Tuple[float, ...]]):
    """
    Class representing implicit ratings of a single user.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # pragma: no cover

    @staticmethod
    def from_im(
        matrix: InteractionMatrix, upper: float = 3.5, lower: float = 0, *args, **kwargs
    ) -> "Implicit":
        """
        Static method for creating an Implicit from an InteractionMatrix.
        The InteractionMatrix provides item IDs and ratings.
        The ratings are converted to implicit ratings by thresholding at upper and lower.

        Args:
            matrix (InteractionMatrix): The InteractionMatrix.
            upper (float, optional): Upper threshold. If the rating is above or equal, the implicit rating is 1. Defaults to 3.5.
            lower (float, optional): Lower threshold. If the rating is lower, the implicit rating is -1. Defaults to 0.

        Returns:
            Implicit: The resulting Implicit.
        """
        res = {}
        for interaction in matrix:
            if interaction[2] >= upper:
                val = 1
            elif interaction[2] < lower:
                val = -1
            else:
                val = 0
            res[int(interaction[1])] = (val, interaction[3])
        return Implicit(res)


class Sequential:
    """
    Class representing sequential ratings of a single user.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def from_im(matrix: InteractionMatrix) -> "Sequential":
        """
        Static method for creating a Sequential from an InteractionMatrix.
        The InteractionMatrix provides item IDs and ratings and timestamps.
        The resulting Sequential is sorted by timestamp.

        Args:
            matrix (InteractionMatrix): The InteractionMatrix.

        Returns:
            Sequential: The resulting Sequential.
        """
        pass


class FeatureVector(np.ndarray):
    """
    numpy.ndarray containing item features.
    """

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    @staticmethod
    def from_im(
        matrix: InteractionMatrix, dataset: FeatureDataset, *args, **kwargs
    ) -> "FeatureVector":
        """
        Static method for creating a FeatureVector from an InteractionMatrix and a FeatureDataset.
        It is created by creating a linear combination of the idividual feature vectors of the items in the
        InteractionMatrix, using the ratings as the coefficients.

        Args:
            matrix (InteractionMatrix): The InteractionMatrix object. Used for gathering item IDs and ratings.
            dataset (FeatureDataset): The FeatureDataset object. Contains the feature vectors of all the items.

        Returns:
            FeatureVector: The resulting FeatureVector.
        """
        vector = np.zeros(dataset.n_features)
        for interaction in matrix.data:
            vector += dataset[interaction[1]] * interaction[2]
        return FeatureVector(vector)


class Recommendation(List[Tuple[Any, float]]):
    """
    List of tuples where the first element is the item ID and the second element is the score.
    """

    def __init__(
        self,
        input: Iterable[Tuple[Any, float]] | Iterable[float] | np.ndarray,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)  # pragma: no cover

        if type(input) is Iterable[Tuple[Any, float]]:
            for item_id, score in input:
                self.append((item_id, score))
        elif type(input) in (np.ndarray, Iterable[float]):
            for i, score in enumerate(input):
                self.append((i, score))
        else:
            raise ValueError("Input must be an iterable of tuples or floats.")

        # sort self by score
        self.sort(key=lambda x: x[1], reverse=True)

    @property
    def ids(self) -> List[int]:
        """
        List of the item IDs.

        Returns:
            List[int]: The resulting list of IDs.
        """
        return [x[0] for x in self]

    @property
    def scores(self) -> List[float]:
        """
        List of the scores.

        Returns:
            List[float]: The resulting list of scores.
        """
        return [x[1] for x in self]


class ResponseDict(dict[str, Any]):
    """
    Dictionary representing a the response to be sent as a JSON
    """

    pass


class ConversionRegistry:
    """
    Static class for converting between different data types.
    """

    _conversions = {
        (InteractionMatrix, Explicit): Explicit.from_im,
        (InteractionMatrix, Implicit): Implicit.from_im,
        (InteractionMatrix, FeatureVector): FeatureVector.from_im,
        (InteractionMatrix, Sequential): Sequential.from_im,
        (Explicit, InteractionMatrix): InteractionMatrix.from_exp,
        (Sequential, InteractionMatrix): InteractionMatrix.from_seq,
    }

    @staticmethod
    def convert(from_type: type, to_type: type, *args, **kwargs) -> Any:
        """
        Static method that takes in the desired input and output type for the conversion,
        and converts the input value to the output type, if possible.


        Args:
            from_type (type): The input type of the conversion.
            to_type (type): The output type of the conversion.

        Raises:
            ValueError: If the type conversion is impossible.

        Returns:
            Any: The converted
        """
        if from_type == to_type:
            return args[0]
        # try to find a direct conversion
        try:
            conversion_function = ConversionRegistry._conversions[(from_type, to_type)]
        except KeyError:
            # try to find an indirect conversion
            for (c1ft, c1tt), f1 in ConversionRegistry._conversions.items():
                for (c2ft, c2tt), f2 in ConversionRegistry._conversions.items():
                    if c1ft == from_type and c2tt == to_type and c1tt == c2ft:
                        x = f1(*args, **kwargs)
                        args = args[1:]
                        return f2(x, *args, **kwargs)

            raise ValueError(
                f"No direct or indirect conversion from {from_type} to {to_type} exists."
            )
        return conversion_function(*args, **kwargs)
