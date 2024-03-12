"""
Module defining classes used for data handling.
"""

from abc import ABC

import numpy as np
import pandas as pd

from .types import FeatureVector


class IDLookup:
    """
    Simple class for translating raw IDs to inner IDs and vice versa.
    """

    def __init__(self, data: pd.Series) -> None:
        """
        Build the lookup dicts from the given series.

        Parameters:
            data (pd.Series): The input data series.

        Returns:
            None
        """
        data = data.drop_duplicates(keep="first")
        self.raw2inner = {v: k for k, v in enumerate(data)}
        self.inner2raw = {k: v for k, v in enumerate(data)}

    def __len__(self):
        return len(self.raw2inner)


class Dataset(ABC):
    """
    Base class for datasets.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize the class with the given dataframe.

        Args:
            data (pd.DataFrame): The input dataframe.

        Returns:
            None
        """
        self.data = data
        self.lookup = IDLookup(data.index.to_series())

    def __len__(self):
        """
        Returns the length of the data.
        """
        return len(self.data)  # pragma: no cover

    @property
    def shape(self):
        """
        Returns the shape of the data.
        """
        return self.data.shape  # pragma: no cover

    @property
    def inner2raw(self):
        """
        Dictionary that converts inner IDs to raw IDs.
        """
        return self.lookup.inner2raw

    @property
    def raw2inner(self):
        """
        Dictionary that converts raw IDs to inner IDs.
        """
        return self.lookup.raw2inner

    def train_test_split(self) -> tuple["Trainset", "Trainset"]:
        """
        Splits the data into train and test sets and returns the Trainset objects for each set.
        """
        train, test = np.split(self.data, [int(0.8 * len(self.data))])
        return Trainset(train), Trainset(test)


class FeatureDataset(Dataset):
    """
    Class for feature data.
    The index should be used for item identifiers.
    The columns represent features.
    """

    def __init__(
        self,
        data: pd.DataFrame | pd.Series,
        features: list | str | None = None,
    ) -> None:
        if type(data) == pd.Series:
            self.lookup = IDLookup(data)

        elif type(data) == pd.DataFrame:
            super().__init__(data)

            # if no features are provided, get all numeric columns and drop columns containing NaN
            if features is None:
                self.features = self.data.select_dtypes(include="number")
                self.features = self.features.dropna(axis=1)
            # subset the features
            if type(features) == list:
                self.features = self.data[features]

            # if its a regex
            if type(features) == str:
                self.features = self.data.filter(regex=features, axis=1)

            if self.features is None:
                raise ValueError("features parameter must be provided")

            if type(self.features) is pd.DataFrame:
                self.features = self.features.to_numpy()

        else:
            raise ValueError("data must be a DataFrame or Series")

    def __getitem__(self, idx) -> FeatureVector:
        return FeatureVector(self.features[idx])


class UtilityMatrix(Dataset):
    """
    Class for handling user-item interaction data.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        user_column: str | int,
        item_column: str | int,
        rating_column: str | int,
        timestamp_column: str | None = None,
    ) -> None:
        """
        Initializes the class with the provided data and column names, and optionally renames the columns for standardization.

        Args:
            data (pd.DataFrame): The input DataFrame.
            user_column (str): The name of the column containing user identifiers.
            item_column (str): The name of the column containing item identifiers.
            rating_column (str): The name of the column containing ratings.
            timestamp_column (str, optional): The name of the column containing timestamps. Defaults to None.

        Returns:
            None
        """
        self.data = data

        user_data = data[user_column]
        item_data = data[item_column]

        self.user_lookup = IDLookup(user_data)
        self.item_lookup = IDLookup(item_data)

        self.data.rename(
            columns={user_column: "user", item_column: "item", rating_column: "rating"},
            inplace=True,
        )

        # convert the raw IDs to inner IDs
        self.data["user"] = self.data["user"].map(self.user_lookup.raw2inner)
        self.data["item"] = self.data["item"].map(self.item_lookup.raw2inner)

        if timestamp_column is not None:
            self.data.rename(
                columns={timestamp_column: "timestamp"},
                inplace=True,
            )

        self._build_matrix()

    def __getitem__(self, idx) -> FeatureVector:
        return FeatureVector(self.features[idx])

    def _build_matrix(self):
        self.matrix = np.ndarray((len(self.user_lookup), len(self.item_lookup)))

        # set the values in the matrix such that the value is the rating
        self.matrix[self.data["user"], self.data["item"]] = self.data["rating"]

    @property
    def features(self) -> np.ndarray:
        """
        Returns the utility matrix as a numpy array.
        """
        return self.matrix


class Trainset:
    """
    Default class for iterating through cases of the dataset.
    You can override the __getitem__ and __len__ methods to customize the behavior.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initializes the class with the given DataFrame while resetting its index.

        Parameters:
            data (pd.DataFrame): The input DataFrame.

        Returns:
            None
        """
        self.data = data.reset_index()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]
