import pandas as pd
import pytest

from ..src.rstk.types import Interaction, InteractionMatrix


@pytest.fixture
def interaction():
    return Interaction(0, 3, 4.2, 102931)


@pytest.fixture
def interaction_list():
    return [
        Interaction(0, 3, 4.2, 102931),
        Interaction(6, 2, 2.1, 102933),
        Interaction(1, 7, 3.7, 102920),
    ]


@pytest.fixture
def interaction_df():
    df = pd.DataFrame(
        {
            "user": [0, 6, 1],
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


def test_interaction(interaction):
    assert interaction.user == 0
    assert interaction.item == 3
    assert interaction.rating == 4.2
    assert interaction.timestamp == 102931


def test_interaction_matrix(matrix_from_iterable, matrix_from_df):
    assert matrix_from_df.data.shape == (3, 4)
    assert matrix_from_iterable.data.shape == (3, 4)


def test_explicit_from_matrix(matrix_from_iterable):
    explicit = matrix_from_iterable.explicit
    assert len(explicit) == 3
    assert explicit[3] == (4.2, 102931)
    assert explicit[7] == (3.7, 102920)
    assert explicit[2] == (2.1, 102933)


def test_implicit_from_matrix(matrix_from_iterable):
    implicit = matrix_from_iterable.implicit
    assert len(implicit) == 3
    assert implicit[3] == (1, 102931)
    assert implicit[7] == (1, 102920)
    assert implicit[2] == (0, 102933)
