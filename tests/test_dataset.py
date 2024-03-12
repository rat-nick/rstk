import pytest


def test_dataset_train_test_split(item_dataset):
    train, test = item_dataset.train_test_split()

    assert len(train) / len(test) == pytest.approx(4, abs=0.01)
