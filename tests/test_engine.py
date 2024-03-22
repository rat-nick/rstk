import pytest

from ..src.rstk.data.types import Explicit, Recommendation
from ..src.rstk.engine import CBSEngine
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


def test_cbsengine(cbs_engine):
    assert cbs_engine is not None


def test_cbsengine_recommend(cbs_engine, explicit_ratings):
    out = cbs_engine.recommend(explicit_ratings)
    assert type(out) == cbs_engine.output_type
