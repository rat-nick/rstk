import os
import pickle
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class KNN:
    """Content based item recommendation system using the k-nearest neighbors algorithm"""

    def __init__(
        self,
        data: pd.DataFrame = None,
        id_column: str = None,
        feature_columns: List[str] = None,
        feature_prefix: str = "ftr",
    ):
        """
        Initialize the KNN class.

        Parameters
        ----------
        data : pd.DataFrame
            The data frame containing the items and their features.
        id_column : str
            The name of the column in the data frame that contains the item IDs.
        feature_prefix : str, optional
            The prefix of the columns in the data frame that contain the features, by default "ftr".
        feature_columns : List[str], optional
            A list of the names of the columns in the data frame that contain the features, by default None. If None, all columns with the specified prefix are used.

        Raises
        ------
        ValueError
            If no features are specified and no columns with the specified prefix are found.
        """
        self.data = data

        if id_column is not None:
            self.data = self.data.set_index(id_column)

        self.translate_ids()

        self.init_features(feature_columns, feature_prefix)

    def init_features(self, feature_columns, feature_prefix):
        """
        Initializes the features based on the provided feature columns and prefix.

        Parameters:
            feature_columns (list): List of feature columns to initialize the features.
            feature_prefix (str): Prefix to filter the feature columns.

        Raises:
            ValueError: If no features are selected.

        Returns:
            None
        """
        if feature_columns is None:
            self.features = self.data.filter(regex=f"^{feature_prefix}*")
        else:
            self.features = self.data[feature_columns]

        if len(self.features.columns) == 0:
            raise ValueError("No features selected")

        # translate the features to a numpy array
        self.features = self.features.to_numpy()

    def translate_ids(self):
        """
        Translates the raw item IDs into unique inner IDs and vice versa.

        The inner IDs are used to represent the items in the recommendation model, while the raw IDs
        are used to retrieve the original data from the database.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # dict where the key is the raw item id and the value is a unique integer for each item
        self.raw2innerID = {}
        # dict where the key is the unique integer for each item and the value is the raw item id
        self.inner2rawID = {}
        for id, _ in self.data.iterrows():
            if id not in self.raw2innerID.keys():
                innerID = len(self.raw2innerID)
                self.raw2innerID[id] = innerID
                self.inner2rawID[innerID] = id

    def get_similar_items(
        self,
        user_profile: np.array,
        k: int = 10,
    ) -> List[any]:
        """
        Get the k most similar items to the given user profile using cosine similarity.

        Args:
            user_profile (np.array): The user profile for which to find similar items.
            k (int): The number of similar items to return. Defaults to 10.

        Returns:
            List[any]: A list of the k most similar items to the user profile.
        """
        if user_profile is None:
            raise ValueError("Must provide the user profile")

        if type(user_profile) is not np.array:
            user_profile = np.array(user_profile)

        # calculate the similarity of all items to the user profile
        similarity = cosine_similarity(user_profile.reshape(1, -1), self.features)

        # get the indicies of the k most similar items
        indicies = np.argsort(similarity, axis=1)[0][::-1]

        # translate the indicies into raw IDs
        return [self.inner2rawID[i] for i in indicies][:k]

    def serialize(self, path: str):
        """
        Serialize the model to a file.

        Args:
            path (str): The file path to serialize the object to.

        Returns:
            None
        """

        # check if its the current directory
        if "/" not in path:
            path = "./" + path
        # check if the path exists and if not create it
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def deserialize(path: str) -> "KNN":
        """
        Deserialize a KNN model from the given path using pickle.

        Args:
            path (str): The path to the serialized KNN model.

        Returns:
            KNN: The deserialized KNN model.
        """
        with open(path, "rb") as f:
            return pickle.load(f)
