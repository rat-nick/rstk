import numpy as np
import pytest

from ..src.rstk.model import SimilarityBased


@pytest.fixture
def similarity_based_model(item_dataset) -> SimilarityBased:
    model = SimilarityBased()
    model.fit(item_dataset)
    return model


def test_forward(similarity_based_model):
    feature_vector = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    output = similarity_based_model.forward(feature_vector)

    assert output[:5].tolist() == [28, 230, 253, 23, 575]


def test_fit(similarity_based_model, item_dataset):
    similarity_based_model.fit(item_dataset)

    assert similarity_based_model.data is not None
