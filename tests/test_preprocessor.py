import numpy as np
import pandas as pd
import pytest

from ..src.rstk.preprocess import Preprocessor


@pytest.fixture
def user_dataframe_with_nan(user_dataframe):
    data = user_dataframe
    # randomly set 100 values from the occupation column to missing
    data.loc[np.random.choice(data.index, 100), "occupation"] = np.nan
    # randomly set 100 values from the age column to missing
    data.loc[np.random.choice(data.index, 100), "age"] = np.nan

    return data


@pytest.fixture
def item_preprocessor(item_dataframe):
    return Preprocessor(df=item_dataframe)


@pytest.fixture
def user_preprocessor(user_dataframe_with_nan):
    return Preprocessor(df=user_dataframe_with_nan)


def test_preprocessor_with_path():
    instance = Preprocessor(path="data/dataset.csv", delimiter="|")
    assert type(instance.data) == pd.DataFrame


def test_preprocssor_with_dataframe(item_dataframe):
    instance = Preprocessor(df=item_dataframe)
    assert type(instance.data) == pd.DataFrame


def test_onehot_encode(user_preprocessor):
    # get all unique values of the occupation column
    unique_occupations = user_preprocessor.data["occupation"].unique()

    user_preprocessor.onehot_encode(columns=["gender", "occupation"])

    # grab a reference to the data for easy access
    data = user_preprocessor.data

    columns = list(data.columns)

    assert "ftr_M" in columns
    assert "ftr_F" in columns

    for occ in unique_occupations:
        assert f"ftr_{occ}" in columns


def test_normalize_linear(user_preprocessor):
    user_preprocessor.handle_missing_values()
    user_preprocessor.normalize(["age"], methods=["linear"])

    # grab a reference to the data for easy access
    data = user_preprocessor.data

    assert (1 >= data["age"]).all()
    assert (0 <= data["age"]).all()


def test_normalize_z_score(user_preprocessor):
    user_preprocessor.handle_missing_values()
    user_preprocessor.normalize(["age"], methods=["z-score"])

    # grab a reference to the data for easy access
    data = user_preprocessor.data

    assert (4 >= data["age"]).all()
    assert (-4 <= data["age"]).all()


def test_dropna(user_preprocessor):
    # grab a reference to the data for easy access
    data = user_preprocessor.data

    start_length = len(data)

    # randomly set 100 values from the occupation column to missing
    user_preprocessor.data.loc[np.random.choice(data.index, 100), "occupation"] = np.nan
    # count the number of rows containing at least 1 missing value
    num_missing = data.isnull().any(axis=1).sum()

    user_preprocessor.handle_missing_values(strategy="drop")

    assert num_missing + len(user_preprocessor.data) == start_length


def test_fillna(user_preprocessor):
    mode = user_preprocessor.data["occupation"].mode()[0]
    mean = user_preprocessor.data["age"].mean()

    # get the indices of the missing values for occupation
    indices = user_preprocessor.data[
        user_preprocessor.data["occupation"].isnull()
    ].index

    user_preprocessor = user_preprocessor.fillna("occupation", "mode")

    assert (user_preprocessor.data.loc[indices, "occupation"] == mode).all()

    # get the indices of the missing values for occupation
    indices = user_preprocessor.data[user_preprocessor.data["age"].isnull()].index

    user_preprocessor.fillna("age", "mean")

    assert (user_preprocessor.data.loc[indices, "age"] == mean).all()


def test_multilabel_binarize(item_preprocessor):
    item_preprocessor.multilabel_binarize(["tags"])
    data = item_preprocessor.data

    assert "ftr_T1" in data.columns
    assert "ftr_T2" in data.columns
    assert "ftr_T3" in data.columns
    assert "ftr_T4" in data.columns
    assert "ftr_T5" in data.columns


def test_select_features_with_column_names(user_preprocessor):
    data = user_preprocessor.select_features(columns=["age", "gender"])
    assert "age" in data.columns
    assert "gender" in data.columns
    assert "occupation" not in data.columns


def test_select_features_with_column_slices(user_preprocessor):
    data = user_preprocessor.select_features(columns=[slice(0, 2), slice(3, 6)])
    assert "age" in data.columns
    assert "gender" in data.columns
    assert "zip code" in data.columns
    assert "occupation" not in data.columns


def test_select_features_with_regex(user_preprocessor):
    data = user_preprocessor.select_features(regex=".*e.*")
    assert "age" in data.columns
    assert "zip code" in data.columns
    assert "gender" in data.columns
    assert "occupation" not in data.columns


def test_simple_preprocessing_chain(item_preprocessor):

    data = (
        item_preprocessor.handle_missing_values(strategy="mean")
        .multilabel_binarize(["tags"])
        .normalize(["price"], methods=["z-score"])
        .onehot_encode(columns=["category"])
        .select_features(regex="^ftr.*", columns=["price"])
    )

    assert "price" in data.columns
    assert "ftr_T1" in data.columns
    assert "ftr_T2" in data.columns
    assert "ftr_T3" in data.columns
    assert "ftr_T4" in data.columns
    assert "ftr_T5" in data.columns
    assert "ftr_X" in data.columns
    assert "ftr_Y" in data.columns
    assert "ftr_Z" in data.columns
