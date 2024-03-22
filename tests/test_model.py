import numpy as np
import pytest

from ..src.rstk.data.types import Recommendation


def test_forward(similarity_based_model):
    feature_vector = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    output: Recommendation = similarity_based_model.forward(feature_vector)
    output = output.ids[:5]
    assert output == [28, 230, 23, 253, 398]


def test_fit(similarity_based_model, item_dataset):
    similarity_based_model.fit(item_dataset)

    assert similarity_based_model.data is not None
