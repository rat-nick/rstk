import os
import pickle
from typing import Dict, List

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

    def get_most_similar(self, vector: np.array):
        """
        Calculate the most similar items to the input vector in the feature space.

        Parameters:
            vector (np.array): The feature vector for which to find the most similar items.
            k (int): The number of most similar items to return.

        Returns:
            list: A list of the raw IDs of the most similar items to the input vector.
        """

        similarity = cosine_similarity(vector.reshape(1, -1), self.features)
        indicies = np.argsort(similarity, axis=1)[0][::-1]
        return [self.inner2rawID[i] for i in indicies]

    def get_recommendations(
        self,
        profile: np.array = None,
        ratings: Dict[any, float] = None,
        preference: List[any] = None,
        k: int = 10,
    ) -> List[any]:
        """
        Get recommendations based on the given user profile, user ratings, or user preference.

        Parameters:
            profile (np.array, optional): The user profile in feature space. Defaults to None.
            ratings (Dict[any, float], optional): The user ratings as a dictionary mapping raw item IDs to ratings. Defaults to None.
            preference (List[any], optional): The user preference as a list of raw item IDs. Defaults to None.
            k (int, optional): The number of recommendations to return. Defaults to 10.

        Returns:
            List[any]: A list of raw IDs of recommended items.

        Raises:
            ValueError: If other than one of profile, ratings, or preference is provided.
        """
        if sum(p is not None for p in [profile, ratings, preference]) != 1:
            raise ValueError(
                "Only one of user_profile, user_ratings, or user_preference can be provided."
            )

        vector = None

        if profile is not None:
            vector = np.array(profile)
            return self.get_most_similar(vector)[:k]

        if ratings is not None:
            raw_ids = list(ratings.keys())
            inner_ids = [self.raw2innerID[item] for item in ratings.keys()]
            ratings = np.array(list(ratings.values())).reshape(1, -1)
            vector = ratings.dot(self.features[inner_ids])
            most_similar = self.get_most_similar(vector)
            most_similar = [x for x in most_similar if x not in raw_ids]
            return most_similar[:k]

        if preference is not None:
            inner_ids = [self.raw2innerID[item] for item in preference]
            vector = np.sum(self.features[inner_ids], axis=0)
            most_similar = self.get_most_similar(vector)
            most_similar = [x for x in most_similar if x not in preference]
            return most_similar[:k]

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
