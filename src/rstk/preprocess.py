from typing import List, Literal

import pandas as pd

z_score = lambda x, mean, std: (x - mean) / std
linear = lambda x, min, max: (x - min) / (max - min)


class Preprocessor:
    """
    The basic class used for data preprocessing. It can be used for loading the dataset or can be used with an existing dataframe.
    Contains a variety of standard methods for data preprocessing that should be called with method chaining.

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

    def __init__(self, path: str = None, df: pd.DataFrame = None, delimiter: str = ","):
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

    def _fillna(self, strategy: Literal["mean", "median", "mode"] = "drop"):
        """
        Function used for filling missing values using the given strategy

        Parameters
        ----------
        strategy : Literal['mean', 'median', 'mode'], optional
            The strategy to be used, by default "drop"
        """
        for col in self.data.columns:
            fill_val = self._determine_fill_value(col, strategy)
            self.data[col] = self.data[col].fillna(fill_val)

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
                fill_val == self.data[col].mode()

        return fill_val

    def normalize(
        self,
        normalization_columns: List[str] = [],
        methods: List[Literal["linear", "z-score"]] = None,
    ) -> "Preprocessor":
        """
        Performs column-wise normalization of the data using the specified methods.

        Parameters
        ----------
        normalization_columns : List[str], optional
            A list of columns to be normalized, by default []
        methods : List[Literal['linear', 'z, optional
            A list od normalization methods to be performed on the given columns, respectively, by default None

        Returns
        -------
        Preprocessor
            The preprocessor instance

        Raises
        ------
        ValueError
            If the lenght of the methods arguments doesn't match the length of the normalization_columns
        """

        if len(normalization_columns) > 0 and methods == None:
            methods = len(normalization_columns) * ["linear"]
        if len(methods) != len(normalization_columns):
            raise ValueError("Method and column lists must be of equal length.")

        for col, meth in zip(normalization_columns, methods):
            self._normalize_column(col, meth)

        return self

    def _normalize_column(self, col, meth):
        """
        Performs the given normalization method on the given column

        Parameters
        ----------
        col : _type_
            The column identifier
        meth : _type_
            The method to be performed
        """
        if meth == "z-score":
            # perform z-score normalization
            self.data[col] = z_score(
                self.data[col], self.data[col].mean(), self.data[col].std()
            )

        elif meth == "linear":
            # perform linear normalization
            self.data[col] = linear(
                self.data[col], self.data[col].min(), self.data[col].max()
            )

    def onehot_encode(self, columns: List[str] = []) -> "Preprocessor":
        # TODO change parameter to be a column range type
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
        self.data = pd.get_dummies(self.data, columns=columns, prefix="ftr")
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

    def select_features(
        self,
        columns: List[str] | str = None,
        regex: str = None,
    ) -> pd.DataFrame:
        # TODO add support for ranges and single columns
        """
        Selects the given features using a column range and regex.

        Args:
            columns (List[str] | str, optional): List of column indentifiers or ranges. Defaults to None.
            regex (str, optional): The regex to be used for selecting columns. Defaults to None.

        Raises:
            ValueError: If no parameters are provided

        Returns:
            pd.DataFrame: The resulting dataframe
        """

        df = pd.DataFrame()

        if regex is not None:
            df = self.data.filter(regex=regex, axis=1)

        if type(columns) == str:
            ranges = columns.split(",")
            for range in ranges:
                start, end = range.split(":")
                if start == "":
                    start = 0
                if end == "":
                    end = len(self.data.columns)
                start, end = int(start), int(end)
                df = pd.concat([df, self.data.iloc[:, start:end]], axis=1, sort=False)

        if type(columns) == list and all(type(x) == str for x in columns):
            df = pd.concat([df, self.data.loc[:, columns]], axis=1, sort=False)

        if len(df.columns) > 0:
            return df

        raise ValueError("Columns must be either a string or a list of strings")
