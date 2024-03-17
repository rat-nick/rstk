"""
Module defining classes used for data preprocessing.
"""

from typing import List, Literal

import pandas as pd

from .data import FeatureSelector

z_score = lambda x, mean, std: (x - mean) / std
linear = lambda x, min, max: (x - min) / (max - min)


class Preprocessor(FeatureSelector):
    """
    The basic class used for data preprocessing. It can be used for loading the dataset or can be used with an existing dataframe.
    Contains a variety of standard methods for data preprocessing that should be called with method chaining.

    Attr:
        data (pd.DataFrame): The dataframe used for preprocessing
        fs (FeatureSelector): The feature selector

    Example::

        preprocessor = Preprocessor(path="dataset.csv")
        data = (
            preprocessor
            .handle_missing_values()
            .multilabel_binarize(["release", "genres"])
            .normalize(["price", "releaseYear"], methods=["z-score", "linear"])
            .select_features(regex="^ftr_.*")
        )
    """

    def __init__(
        self,
        path: str | None = None,
        df: pd.DataFrame | None = None,
        delimiter: str = ",",
    ):
        """
        Initializes the object with optional path, DataFrame, and delimiter parameters.

        Parameters:
            path (str): The path to the file to load.
            df (pd.DataFrame): The DataFrame to initialize the object with.
            delimiter (str): The delimiter for the file, defaults to ",".

        Returns:
            None
        """

        if path != None:
            self.load(path, delimiter)
        elif df is not None:
            self.data = df

        super().__init__(self.data)

    def load(self, path: str, delimiter: str = ",") -> "Preprocessor":
        """
        Loads the dataset from the given path.

        Parameters
        ----------
        path : str
            The path to the dataframe
        delimiter : str, optional
            string used for splitting, by default ","

        Returns
        -------
        Preprocessor
            The preprocessor instance
        """
        self.data = pd.read_csv(path, delimiter=delimiter, header=0, engine="python")
        return self

    def handle_missing_values(
        self, strategy: Literal["drop", "mean", "median", "mode"] = "drop"
    ) -> "Preprocessor":
        """
        Performs standard handling of missing values with a variety of strategies.

        Parameters
        ----------
        strategy : Literal['drop, 'mean', 'median', 'mode'], optional
            The strategy to be used when handling missing values, by default "drop"

        Returns
        -------
        Preprocessor
            The preprocessor instance
        """
        if strategy == "drop":
            self.data.dropna(inplace=True)
        else:
            self._fillna(strategy=strategy)

        return self

    def _fillna(self, strategy: Literal["mean", "median", "mode"] = "mean"):
        """
        Function used for filling missing values using the given strategy

        Parameters
        ----------
        strategy : Literal['mean', 'median', 'mode'], optional
            The strategy to be used, by default "mean"
        """
        for col in self.data.columns:
            fill_val = self._determine_fill_value(col, strategy)
            self.data[col] = self.data[col].fillna(fill_val)

    def fillna(
        self, column: str, strategy: Literal["mean", "median", "mode"] = "mean"
    ) -> "Preprocessor":
        """
        Fills missing values in a specific column using the specified strategy and returns the updated Preprocessor object.

        Args:
            column (str): The name of the column to fill missing values for.
            strategy (Literal["mean", "median", "mode"]): The strategy to determine the fill value, by default "mean".

        Returns:
            Preprocessor: The updated Preprocessor object.
        """
        fill_val = self._determine_fill_value(column, strategy)
        self.data[column] = self.data[column].fillna(fill_val)
        return self

    def _determine_fill_value(self, col, strategy):
        dtype = self.data[col].dtype

        fill_val = 0
        if dtype == "object":
            fill_val = self.data[col].mode()[0]
        if dtype in ["int64", "float64"]:
            if strategy == "mean":
                fill_val = self.data[col].mean()
            if strategy == "median":
                fill_val = self.data[col].median()
            if strategy == "mode":
                fill_val = self.data[col].mode()[0]

        return fill_val

    def normalize(
        self, column: str, method: Literal["z-score", "linear"] = "linear"
    ) -> "Preprocessor":
        """
        Performs normalization on the given column using the specified method.

        Args:
            column (str): The name of the column to normalize.
            method (str): The normalization method to use, either "z-score" or "linear".

        Returns:
            Preprocessor: The updated Preprocessor object.

        Raises:
            ValueError: If the specified method is not supported.
        """
        if method == "z-score":
            # perform z-score normalization
            self.data[column] = z_score(
                self.data[column], self.data[column].mean(), self.data[column].std()
            )

        elif method == "linear":
            # perform linear normalization
            self.data[column] = linear(
                self.data[column], self.data[column].min(), self.data[column].max()
            )

        else:
            raise ValueError("Invalid normalization method.")

        return self

    def onehot_encode(self, columns: List[str]) -> "Preprocessor":
        """
        Performs one-hot encoding on the given columns. The resulting colums have the prefix `ftr_`

        Parameters
        ----------
        columns : List[str], optional
            The list of column identifiers, by default []

        Returns
        -------
        Preprocessor
            The preprocessor instance
        """
        self.data = pd.get_dummies(
            self.data, columns=columns, prefix="ftr", dummy_na=True
        )
        return self

    def multilabel_binarize(
        self, multilabel_columns: List[str] = [], sep="|"
    ) -> "Preprocessor":
        # TODO rename parameter to `columns`
        """
        Performs multilabel binarization on the given columns. The resulting colums have the prefix `ftr_`

        Parameters
        ----------
        multilabel_columns : List[str], optional
            The list of column identifiers, by default []
        sep : str, optional
            The separator to be used when splitting labels, by default "|"

        Returns
        -------
        Preprocessor
            The preprocessor instance
        """
        for col in multilabel_columns:
            binarized_labels = self.data[col].str.get_dummies(sep).add_prefix("ftr_")
            self.data.drop(columns=col, inplace=True)
            self.data = pd.concat([self.data, binarized_labels], axis=1)
        return self
