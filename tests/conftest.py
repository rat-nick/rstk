import pandas as pd
import pytest

from ..src.rstk.data.dataset import UtilityMatrix
from ..src.rstk.data.types import (
    Explicit,
    FeatureDataset,
    Interaction,
    InteractionMatrix,
    Recommendation,
)
from ..src.rstk.engine import CBSEngine, CFSEngine
from ..src.rstk.model import SimilarityBased
from ..src.rstk.preprocess import Preprocessor


@pytest.fixture
def interaction():
    return Interaction(4, 3, 4.2, 102931)


@pytest.fixture
def interaction_list():
    return [
        Interaction(4, 3, 4.2, 102931),
        Interaction(4, 2, 2.1, 102933),
        Interaction(4, 7, 3.7, 102920),
    ]


@pytest.fixture
def interaction_df():
    df = pd.DataFrame(
        {
            "user": [4, 4, 4],
            "item": [3, 2, 7],
            "rating": [4.2, 4.2, 4.2],
            "timestamp": [102931, 102933, 102920],
        }
    )
    return df


@pytest.fixture
def matrix_from_iterable(interaction_list):
    return InteractionMatrix(interactions=interaction_list)


@pytest.fixture
def matrix_from_df(interaction_df):
    return InteractionMatrix(dataframe=interaction_df)


@pytest.fixture
def similarity_based_model(item_dataset) -> SimilarityBased:
    model = SimilarityBased()
    model.fit(item_dataset)
    return model


@pytest.fixture
def explicit_ratings(matrix_from_df):
    return Explicit.from_im(matrix_from_df)


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

    return dataset


@pytest.fixture
def user_dataframe():
    df = pd.read_csv("data/ml-100k/u.user", delimiter="|", index_col="user id")
    return df


@pytest.fixture
def user_dataset(user_dataframe):
    dataset = FeatureDataset(user_dataframe)

    assert hasattr(dataset, "data")

    return dataset


@pytest.fixture
def utility_matrix(user_item_interaction_dataframe):
    df = user_item_interaction_dataframe
    dataset = UtilityMatrix(df, user_column=0, item_column=1, rating_column=2)

    assert hasattr(dataset, "data")
    assert hasattr(dataset, "matrix")
    assert hasattr(dataset, "user_lookup")
    assert hasattr(dataset, "item_lookup")

    return dataset


@pytest.fixture
def cbs_engine(item_dataset):
    model = SimilarityBased()

    engine = CBSEngine(
        item_dataset,
        model,
        input_type=Explicit,
        output_type=Recommendation,
    )
    return engine


@pytest.fixture
def cfbs_engine(utility_matrix):
    model = SimilarityBased()
    engine = CFSEngine(utility_matrix, model)
    return engine
