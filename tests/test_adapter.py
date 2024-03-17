from re import A

import numpy as np
import pytest

from ..src.rstk.adapter import Adapter, AdapterBuilder, FeatureAdapter, IDAdapter
from ..src.rstk.data import FeatureVector


@pytest.fixture
def adapter():
    return Adapter()


@pytest.fixture
def id_adapter(user_dataset):
    return IDAdapter(user_dataset)


@pytest.fixture
def feature_adapter(item_dataset):
    return FeatureAdapter(item_dataset)


@pytest.fixture
def composite_raw2feature_adapter(id_adapter, feature_adapter):
    return AdapterBuilder().add(id_adapter).add(feature_adapter).build()


def test_adapter(adapter):
    assert adapter("foo") == "foo"
    assert adapter(1) == 1

    assert adapter.backward("foo") == "foo"
    assert adapter.backward(1) == 1


def test_id_adapter(id_adapter):
    assert id_adapter(13) == 12
    assert id_adapter.backward(12) == 13


def test_feature_adapter(feature_adapter):
    expected = FeatureVector([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    recieved = feature_adapter(9)
    assert np.array_equal(recieved, expected)


def test_composite_raw2feature_adapter(composite_raw2feature_adapter):
    expected = FeatureVector([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    recieved = composite_raw2feature_adapter(10)
    assert np.array_equal(recieved, expected)


def test_adapter_inversion():
    adapter = Adapter(function=lambda x: x * 10, inverse_function=lambda x: x / 10)
    inverse = adapter.inverted

    assert adapter(inverse(5)) == 5
    assert inverse(adapter(5)) == 5


def test_composite_adapter_inversion():
    adapter1 = Adapter(function=lambda x: x * 10, inverse_function=lambda x: x / 10)
    adapter2 = Adapter(function=lambda x: x + 5, inverse_function=lambda x: x - 5)

    adapter1.next = adapter2
    inverse = adapter1.inverted
    assert inverse(5) == 5


def test_adapter_chain():
    adapter1 = Adapter(function=lambda x: x * 10)
    adapter2 = Adapter(function=lambda x: x + 5)

    assert adapter1(adapter2(5)) == 100
    assert adapter2(adapter1(5)) == 55

    adapter1.next = adapter2
    assert adapter1(5) == 55
    adapter1.next = None

    adapter2.next = adapter1
    assert adapter2(5) == 100
