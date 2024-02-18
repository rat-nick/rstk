import os

import numpy as np
import pandas as pd
import pytest

from ..src.algo.content_based.knn import KNN


@pytest.fixture
def feature_data():
    # generate 10 rows and 5 columns of binary data
    data = {
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "ftr_1": [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        "ftr_2": [0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
        "ftr_3": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "ftr_4": [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        "ftr_5": [1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
    }
    df = pd.DataFrame(data)
    return df


def test_knn_constructors(feature_data):
    knn = KNN(data=feature_data, id_column="id")

    assert knn.features.shape == (12, 5)

    knn = KNN(
        data=feature_data,
        id_column="id",
        feature_columns=["ftr_1", "ftr_3"],
    )

    assert knn.features.shape == (12, 2)

    knn = KNN(data=feature_data)

    assert knn.features.shape == (12, 5)

    with pytest.raises(ValueError):
        knn = KNN(
            data=feature_data,
            id_column="id",
            feature_prefix="nonexistent",
        )


@pytest.fixture
def knn(feature_data):
    return KNN(data=feature_data, id_column="id")


@pytest.fixture
def correct_user_profile():
    return np.array([1, 1, 0, 1, 0])


def test_get_similar_items(knn, correct_user_profile):
    recs = knn.get_similar_items(correct_user_profile, 3)
    assert recs == [9, 6, 4]


def test_get_similar_items_with_list_user_profile(knn):
    user_profile = [1, 1, 0, 1, 0]
    recs = knn.get_similar_items(user_profile, 3)
    assert recs == [9, 6, 4]


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
