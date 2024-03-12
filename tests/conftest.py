import pandas as pd
import pytest

from ..src.rstk.adapter import Adapter
from ..src.rstk.dataset import ItemDataset, UtilityMatrix
from ..src.rstk.engine import CBSEngine, CFBSEngine
from ..src.rstk.model import SimilarityBased


@pytest.fixture
def item_dataframe():
    df = pd.read_csv("data/ml-100k/u.item", delimiter="|", index_col="movie id")
    return df


@pytest.fixture
def item_dataset(item_dataframe):
    dataset = ItemDataset(item_dataframe)

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
    dataset = ItemDataset(user_dataframe)

    assert hasattr(dataset, "data")
    assert hasattr(dataset, "inner2raw")
    assert hasattr(dataset, "raw2inner")

    return dataset


@pytest.fixture
def utility_matrix():
    df = pd.read_csv("data/ml-100k/u.data", delimiter="\t")
    return UtilityMatrix(df)


@pytest.fixture
def cbs_engine(item_dataset):
    model = SimilarityBased()
    engine = CBSEngine(model, item_dataset)
    return engine


@pytest.fixture
def cfbs_engine(utility_matrix):
    model = SimilarityBased()
    engine = CFBSEngine(model, utility_matrix)
    return engine
