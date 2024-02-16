from typing import List
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class KNN:
    """Content based item recommendation system using the k-nearest neighbors algorithm"""

    def __init__(
        self,
        data: pd.DataFrame,
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
        Returns the k most similar items to the user profile based on the cosine similarity of the user profile and the item features.

        Parameters
        ----------
        user_profile : np.array
            The user profile represented as a numpy array of dimensions (n_features,)
        k : int, optional
            The number of most similar items to be returned, by default 10

        Returns
        -------
        List[any]
            A list of the raw IDs of the k most similar items to the user profile

        Raises
        ------
        ValueError
            If the user profile is not provided
        """

        if user_profile is None:
            raise ValueError("Must provide the user profile")

        # calculate the similarity of all items to the user profile
        similarity = cosine_similarity(user_profile.reshape(1, -1), self.features)

        # get the indicies of the k most similar items
        indicies = np.argsort(similarity, axis=1)[0][::-1]

        # translate the indicies into raw IDs
        return [self.inner2rawID[i] for i in indicies][:k]
