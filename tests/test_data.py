import pytest


def test_dataset_train_test_split(item_dataset):
    train, test = item_dataset.train_test_split()

    assert len(train) / len(test) == pytest.approx(4, abs=0.01)


def test_utility_matrix(utility_matrix):
    assert utility_matrix is not None


def test_utlity_matrix_build_matrix(utility_matrix, user_item_interaction_dataframe):
    utility_matrix._build_matrix()

    for user, item, rating, _ in user_item_interaction_dataframe.itertuples(
        index=False
    ):
        assert utility_matrix.matrix[user, item] == rating
