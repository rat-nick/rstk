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

    def load(self, path: str, delimiter: str = ",") -> pd.DataFrame:
        return pd.read_csv(path, delimiter=delimiter, header=0, engine="python")

    def preprocess(
        self,
        missing_value_strategy: Literal["drop", "mean", "median", "mode"] = "drop",
        normalization_columns: List[str] = None,
        normalization_methods: List[Literal["linear", "z-score"]] = None,
        categorical_columns: List[str] = None,
        multilabel_columns: List[str] = None,
        feature_columns: List[str] = None,
    ) -> pd.DataFrame:
        self.handle_missing_values(missing_value_strategy)

        if normalization_columns != None:
            self.normalize(normalization_columns, normalization_methods)

        if categorical_columns != None:
            self.onehot_encode(categorical_columns)

        if multilabel_columns != None:
            self.multilabel_binarize(multilabel_columns)
        # get all the columns that were generated via feature engineering
        res_df = self.data.filter(regex="^ftr_*")

        return res_df.join(self.data[[feature_columns]])

    def handle_missing_values(
        self, strategy: Literal["drop", "mean", "median", "mode"] = "drop"
    ):
        if strategy == "drop":
            self.data.dropna(inplace=True)
        else:
            self.fillna(strategy=strategy)

    def fillna(self, strategy: Literal["mean", "median", "mode"] = "drop"):
        for col in self.data.columns:
            fill_val = self.determine_fill_value(col, strategy)
            self.data[col] = self.data[col].fillna(fill_val)

    def determine_fill_value(self, col, strategy):
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
    ):
        """Normalizes the given columns of the DataFrame using the specified method."""

        if len(normalization_columns) > 0 and methods == None:
            methods = len(normalization_columns) * ["linear"]
        if len(methods) != len(normalization_columns):
            raise ValueError("Method and column lists must be of equal length.")

        for col, meth in zip(normalization_columns, methods):
            self.normalize_column(col, meth)

    def normalize_column(self, col, meth):
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

    def onehot_encode(self, categorical_columns: List[str] = []):
        self.data = pd.get_dummies(self.data, columns=categorical_columns, prefix="ftr")

    def multilabel_binarize(self, multilabel_columns: List[str] = [], sep="|"):
        for col in multilabel_columns:
            binarized_labels = self.data[col].str.get_dummies(sep).add_prefix("ftr_")
            self.data.drop(columns=col, inplace=True)
            self.data = pd.concat([self.data, binarized_labels], axis=1)
        print(self.data)

    def select_features(self, columns: List[str] = []) -> pd.DataFrame:
        pass
