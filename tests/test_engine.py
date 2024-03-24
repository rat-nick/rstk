import pytest

from ..src.rstk.data.types import Explicit, Recommendation
from ..src.rstk.engine import CBSEngine, CFSEngine
from ..src.rstk.model import SimilarityBased


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
def cfs_engine(utility_matrix):
    model = SimilarityBased()
    engine = CFSEngine(
        utility_matrix,
        model,
        input_type=Explicit,
        output_type=Recommendation,
    )

    return engine


def test_cbsengine(cbs_engine):
    assert cbs_engine is not None


def test_cbsengine_recommend(cbs_engine, explicit_ratings):
    out = cbs_engine.recommend(explicit_ratings)
    assert type(out) == cbs_engine.output_type


def test_cfs_engine_recommend(cfs_engine, explicit_ratings):
    out = cfs_engine.recommend(explicit_ratings)
    assert type(out) == cfs_engine.output_type
