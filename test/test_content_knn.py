import os

import numpy as np
import pandas as pd
import pytest

from ..src.algo.content_based.knn import KNN


@pytest.fixture
def feature_data():
    # generate 15 rows of data with 4 columns encoding their IDs in binary
    data = {
        "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "ftr_3": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        "ftr_2": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        "ftr_1": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        "ftr_0": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def movie_data():
    data = pd.read_csv("data/u.item", delimiter="|")
    df = pd.DataFrame()


def test_knn_constructors(feature_data):
    knn = KNN(data=feature_data, id_column="id")

    assert knn.features.shape == (16, 4)

    knn = KNN(
        data=feature_data,
        id_column="id",
        feature_columns=["ftr_1", "ftr_3"],
    )

    assert knn.features.shape == (16, 2)

    knn = KNN(data=feature_data)

    assert knn.features.shape == (16, 4)

    with pytest.raises(ValueError):
        knn = KNN(
            data=feature_data,
            id_column="id",
            feature_prefix="nonexistent",
        )


@pytest.fixture
def knn(feature_data):
    return KNN(data=feature_data, id_column="id")


def test_get_most_similar(knn):
    user_profile = np.array([0, 1, 1, 1])
    recs = knn.get_most_similar(user_profile)
    assert recs == [7, 15, 6, 5, 3, 14, 13, 11, 4, 2, 1, 12, 10, 9, 8, 0]


def test_get_most_similar_items_with_incorrect_user_profile(knn):
    user_profile = ["h", "e", "l", "l", "o"]
    with pytest.raises(ValueError):
        knn.get_recommendations(user_profile)


def test_get_recommendations_with_profile(knn):
    profile = np.array([0, 1, 1, 1])
    recs = knn.get_recommendations(profile=profile, k=3)
    assert recs == [7, 15, 6]


def test_get_recommendations_with_ratings(knn):
    ratings = {6: 1, 1: 1}
    recs = knn.get_recommendations(ratings=ratings, k=3)
    recs == [7, 15, 5]


def test_get_recommendations_with_preference(knn):
    preference = [6, 1]
    recs = knn.get_recommendations(preference=preference, k=3)
    recs == [7, 15, 5]


def test_model_searialization(knn):
    path = "./example.pkl"
    knn.serialize(path)

    assert os.path.exists(path)

    deserialized_knn = KNN.deserialize(path=path)

    assert type(deserialized_knn) == KNN

    os.remove(path)


def test_model_serialization_with_nested_path(knn):
    path = "nested/path.pkl"
    knn.serialize(path)

    assert os.path.exists(path)

    deserialized_knn = KNN.deserialize(path=path)

    assert type(deserialized_knn) == KNN

    os.remove(path)
    os.removedirs(os.path.dirname(path))
