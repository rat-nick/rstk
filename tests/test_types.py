from ..src.rstk.data.types import (
    ConversionRegistry,
    Explicit,
    FeatureVector,
    Implicit,
    InteractionMatrix,
)


def test_interaction(interaction):
    assert interaction.user == 4
    assert interaction.item == 3
    assert interaction.rating == 4.2
    assert interaction.timestamp == 102931


def test_interaction_matrix(matrix_from_iterable, matrix_from_df):
    assert matrix_from_df.data.shape == (3, 4)
    assert matrix_from_iterable.data.shape == (3, 4)


def test_im2exp(matrix_from_iterable):
    exp = ConversionRegistry.convert(InteractionMatrix, Explicit, matrix_from_iterable)
    assert len(exp) == 3
    assert exp[3] == (4.2, 102931)
    assert exp[2] == (2.1, 102933)
    assert exp[7] == (3.7, 102920)


def test_im2imp(matrix_from_iterable):
    imp = ConversionRegistry.convert(InteractionMatrix, Implicit, matrix_from_iterable)
    assert len(imp) == 3
    assert imp[3] == (1, 102931)
    assert imp[2] == (0, 102933)
    assert imp[7] == (1, 102920)


def test_im2fv(matrix_from_df, item_dataset):
    fv: FeatureVector = ConversionRegistry.convert(
        InteractionMatrix, FeatureVector, matrix_from_df, item_dataset
    )
    assert fv.shape == (item_dataset.n_features,)


def test_exp2imp(explicit_ratings):
    imp = ConversionRegistry.convert(Explicit, Implicit, explicit_ratings)
    assert len(imp) == 3
    assert imp[3] == (1, 102931)
    assert imp[2] == (1, 102933)
    assert imp[7] == (1, 102920)
