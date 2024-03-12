"""
Module defining classes used for data handling.
"""

from abc import ABC

import numpy as np
import pandas as pd


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
        self.inner2raw = {}
        self.raw2inner = {}
        self._build_translation_dicts()

    @property
    def shape(self):
        """
        Returns the shape of the data.
        """
        return self.data.shape

    def _build_translation_dicts(self):
        """
        Builds translation dictionaries for inner and raw ids.
        """
        self.inner2raw = {k: v for k, v in enumerate(self.data.index)}
        self.raw2inner = {v: k for k, v in enumerate(self.data.index)}

    def train_test_split(self) -> tuple["Trainset", "Trainset"]:
        """
        Splits the data into train and test sets and returns the Trainset objects for each set.
        """
        train, test = np.split(self.data, [int(0.8 * len(self.data))])
        return Trainset(train), Trainset(test)


class ItemDataset(Dataset):
    """
    Class for item data.
    The index should be used for item identifiers.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        features: list | str = None,
    ) -> None:
        super().__init__(data)

        # if no features are provided, get all numeric columns and drop columns containing NaN
        if features is None:
            self.features = self.data.select_dtypes(include=np.number)
            self.features = self.features.dropna(axis=1)
        # subset the features
        if type(features) == list:
            self.features = self.data[features]

        # if its a regex
        if type(features) == str:
            self.features = self.data.filter(regex=features, axis=1)


class UtilityMatrix(Dataset):
    """
    Class for handling user-item interaction data.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        user_column: str,
        item_column: str,
        rating_column: str,
        timestamp_column: str = None,
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

        self.user_dataset = ItemDataset(data[user_column])
        self.item_dataset = ItemDataset(data[item_column])

        self.data = data

        self.data.rename(
            columns={user_column: "user", item_column: "item", rating_column: "rating"},
            axis=1,
            inplace=True,
        )

        if timestamp_column is not None:
            self.data.rename(
                columns={timestamp_column: "timestamp"},
                axis=1,
                inplace=True,
            )


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
