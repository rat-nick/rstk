import pytest


def test_cbsengine(cbs_engine):
    assert cbs_engine is not None


def test_cbsengine_recommend(cbs_engine):
    out = cbs_engine.recommend()
