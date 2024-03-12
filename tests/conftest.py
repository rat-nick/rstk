import pandas as pd
import pytest
from requests import head

from ..src.rstk.adapter import Adapter
from ..src.rstk.dataset import FeatureDataset, UtilityMatrix
from ..src.rstk.engine import CBSEngine, CFBSEngine
from ..src.rstk.model import SimilarityBased
from ..src.rstk.preprocess import Preprocessor


@pytest.fixture
def user_item_interaction_dataframe():
    df = pd.read_csv("data/ml-100k/u.data", delimiter="\t", header=None)
    return df


@pytest.fixture
def item_dataframe():
    df = pd.read_csv("data/ml-100k/u.item", delimiter="|", index_col="movie id")
    return df


@pytest.fixture
def item_dataset(item_dataframe):
    dataset = FeatureDataset(item_dataframe)

    assert hasattr(dataset, "data")
    assert hasattr(dataset, "inner2raw")
    assert hasattr(dataset, "raw2inner")

    return dataset


@pytest.fixture
def user_dataframe():
    df = pd.read_csv("data/ml-100k/u.user", delimiter="|", index_col="user id")
    return df


@pytest.fixture
def user_dataset(user_dataframe):
    dataset = FeatureDataset(user_dataframe)

    assert hasattr(dataset, "data")
    assert hasattr(dataset, "inner2raw")
    assert hasattr(dataset, "raw2inner")

    return dataset


@pytest.fixture
def utility_matrix(user_item_interaction_dataframe):
    df = user_item_interaction_dataframe
    dataset = UtilityMatrix(df, user_column=0, item_column=1, rating_column=2)

    assert hasattr(dataset, "data")
    assert hasattr(dataset, "matrix")
    assert hasattr(dataset, "user_lookup")
    assert hasattr(dataset, "item_lookup")
    assert "user" in dataset.data.columns
    assert "item" in dataset.data.columns
    assert "rating" in dataset.data.columns
    return dataset


@pytest.fixture
def cbs_engine(item_dataset):
    model = SimilarityBased()
    engine = CBSEngine(item_dataset, model)
    return engine


@pytest.fixture
def cfbs_engine(utility_matrix):
    model = SimilarityBased()
    engine = CFBSEngine(utility_matrix, model)
    return engine
