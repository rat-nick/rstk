from typing import List, Literal

import pandas as pd

z_score = lambda x, mean, std: (x - mean) / std
linear = lambda x, min, max: (x - min) / (max - min)


class Preprocessor:
    def __init__(self, path: str = None, df: pd.DataFrame = None, delimiter: str = ","):
        if path != None:
            self.data = self.load(path, delimiter)
        elif df is not None:
            self.data = df
        else:
            raise ValueError("Must provide either path to data or dataframe object")

    def load(self, path: str, delimiter: str = ",") -> "Preprocessor":
        self.data = pd.read_csv(path, delimiter=delimiter, header=0, engine="python")
        return self

    def handle_missing_values(
        self, strategy: Literal["drop", "mean", "median", "mode"] = "drop"
    ) -> "Preprocessor":
        if strategy == "drop":
            self.data.dropna(inplace=True)
        else:
            self._fillna(strategy=strategy)

        return self

    def _fillna(self, strategy: Literal["mean", "median", "mode"] = "drop"):
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
        """Normalizes the given columns of the DataFrame using the specified method."""

        if len(normalization_columns) > 0 and methods == None:
            methods = len(normalization_columns) * ["linear"]
        if len(methods) != len(normalization_columns):
            raise ValueError("Method and column lists must be of equal length.")

        for col, meth in zip(normalization_columns, methods):
            self._normalize_column(col, meth)

        return self

    def _normalize_column(self, col, meth):
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
        self.data = pd.get_dummies(self.data, columns=columns, prefix="ftr")
        return self

    def multilabel_binarize(
        self, multilabel_columns: List[str] = [], sep="|"
    ) -> "Preprocessor":
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
        """
        Selects and returns specific columns from the data based on the provided column names or regex pattern.

        Args:
            columns (str or List[str]): The column names to select from the data.
            regex (str): A regex pattern to select columns based on their names.

        Returns:
            pd.DataFrame: The selected columns as a pandas DataFrame.

        Raises:
            ValueError: If the columns input is not a valid type.
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
            df = pd.concat([df, self.data.loc[:, columns]])

        if len(df.columns) > 0:
            return df

        raise ValueError("Columns must be either a string or a list of strings")
