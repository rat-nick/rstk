from textwrap import fill

import numpy as np
import pandas as pd
import pytest

from ..src.rstk.preprocess import Preprocessor


@pytest.fixture
def multilabel_preprocessor():
    df = pd.DataFrame(
        {
            "label": ["T1|T2", "T3", "T2|T4|T1", None, "T5", "T1|T2|T3|T4|T5"],
        }
    )
    return Preprocessor(df=df)


@pytest.fixture
def user_dataframe_with_nan(user_dataframe):
    data = user_dataframe
    # randomly set 100 values from the occupation column to missing
    data.loc[np.random.choice(data.index, 100), "occupation"] = np.nan
    # randomly set 100 values from the age column to missing
    data.loc[np.random.choice(data.index, 100), "age"] = np.nan

    return data


@pytest.fixture
def user_preprocessor(item_dataframe):
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


def test_normalize_with_unkown_method(user_preprocessor):
    with pytest.raises(ValueError):
        user_preprocessor.normalize("age", "unknown")


def test_normalize_linear(user_preprocessor):
    user_preprocessor.handle_missing_values()
    user_preprocessor.normalize("age", "linear")

    # grab a reference to the data for easy access
    data = user_preprocessor.data

    assert (1 >= data["age"]).all()
    assert (0 <= data["age"]).all()


def test_normalize_z_score(user_preprocessor):
    user_preprocessor.handle_missing_values()
    user_preprocessor.normalize("age", "z-score")

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


def test_fillna_mean(user_preprocessor):
    mean = user_preprocessor.data["age"].mean()

    # get the indices of the missing values for age
    indices = user_preprocessor.data[user_preprocessor.data["age"].isnull()].index

    user_preprocessor.fillna("age", "mean")

    assert (user_preprocessor.data.loc[indices, "age"] == mean).all()


def test_fillna_median(user_preprocessor):
    median = user_preprocessor.data["age"].median()

    # get the indices of the missing values for occupation
    indices = user_preprocessor.data[user_preprocessor.data["age"].isnull()].index

    user_preprocessor = user_preprocessor.fillna("age", "median")

    assert (user_preprocessor.data.loc[indices, "age"] == median).all()


def test_fillna_mode(user_preprocessor):
    mode = user_preprocessor.data["age"].mode()[0]

    # get the indices of the missing values for occupation
    indices = user_preprocessor.data[user_preprocessor.data["age"].isnull()].index

    user_preprocessor = user_preprocessor.fillna("age", "mode")

    assert (user_preprocessor.data.loc[indices, "age"] == mode).all()


def test_fillna_nominal(user_preprocessor):
    mode = user_preprocessor.data["occupation"].mode()[0]

    # get the indices of the missing values for occupation
    indices = user_preprocessor.data[
        user_preprocessor.data["occupation"].isnull()
    ].index

    # the strategy should not matter if the column is nominal
    user_preprocessor = user_preprocessor.fillna("occupation", "mean")

    assert (user_preprocessor.data.loc[indices, "occupation"] == mode).all()


def test_multilabel_binarize(multilabel_preprocessor):
    multilabel_preprocessor.multilabel_binarize(["label"])
    data = multilabel_preprocessor.data

    assert "ftr_T1" in data.columns
    assert "ftr_T2" in data.columns
    assert "ftr_T3" in data.columns
    assert "ftr_T4" in data.columns
    assert "ftr_T5" in data.columns


def test_indexer(user_preprocessor):
    res = user_preprocessor[:]

    for col in user_preprocessor.data.columns:
        assert col in res.columns

    res = user_preprocessor[0:2]

    assert "age" in res.columns
    assert "gender" in res.columns
    assert "zip code" not in res.columns

    res = user_preprocessor["age", 1:3, 3]

    for col in user_preprocessor.data.columns:
        assert col in res.columns


def test_indexer_fail(user_preprocessor):
    with pytest.raises(TypeError):
        user_preprocessor[3.14]


def test_simple_preprocessing_chain(user_preprocessor):

    data = (
        user_preprocessor.fillna("age")
        .onehot_encode(["gender", "occupation"])
        .normalize("age", "z-score")[:]
    )

    assert "gender" not in data.columns
    assert "occupation" not in data.columns
    assert "ftr_F" in data.columns
    assert "ftr_M" in data.columns
