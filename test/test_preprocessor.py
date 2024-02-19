import pandas as pd

from src.preprocessor import Preprocessor


def test_preprocessor_with_path():
    instance = Preprocessor(path="data/ml-1m/users.dat", delimiter="::")
    assert type(instance.data) == pd.DataFrame


def test_preprocessor_with_df(user_df):
    instance = Preprocessor(df=user_df)
    assert type(instance.data) == pd.DataFrame


def test_onehot_encode(user_preprocessor):
    user_preprocessor.onehot_encode(categorical_columns=["gender"])

    # grab a reference to the data for easy access
    data = user_preprocessor.data

    assert "ftr_M" in list(data.columns)
    assert "ftr_F" in list(data.columns)
    assert "ftr_U" in list(data.columns)
    assert data["ftr_M"].sum() == 2
    assert data["ftr_F"].sum() == 2
    assert data["ftr_U"].sum() == 1


def test_normalize_linear(user_preprocessor):
    user_preprocessor.normalize(["age"], methods=["linear"])

    # grab a reference to the data for easy access
    data = user_preprocessor.data

    assert (1 >= data["age"]).all()
    assert (0 <= data["age"]).all()


def test_normalize_z_score(user_preprocessor):
    user_preprocessor.normalize(["age"], methods=["z-score"])

    # grab a reference to the data for easy access
    data = user_preprocessor.data

    assert (1.8 >= data["age"]).all()
    assert (-1.5 <= data["age"]).all()


def test_handle_missing_values_by_dropping(item_preprocessor):
    # grab a reference to the data for easy access
    data = item_preprocessor.data

    item_preprocessor.handle_missing_values(strategy="drop")

    assert len(data) == 4


def test_handle_missing_values_with_mean(item_preprocessor):
    data = item_preprocessor.data

    category_mode = data["category"].mode()[0]

    price_mean = data["price"].mean()
    item_preprocessor.handle_missing_values(strategy="mean")

    assert data.loc[6, "price"] == price_mean
    assert data.loc[4, "category"] == category_mode


def test_multilabel_binarize(item_preprocessor):
    item_preprocessor.multilabel_binarize(["tags"])
    data = item_preprocessor.data

    assert "ftr_T1" in data.columns
    assert "ftr_T2" in data.columns
    assert "ftr_T3" in data.columns
    assert "ftr_T4" in data.columns
    assert "ftr_T5" in data.columns


def test_select_features_with_column_names(item_preprocessor):
    data = item_preprocessor.select_features(columns=["category", "price"])
    assert "category" in data.columns
    assert "price" in data.columns


def test_select_features_with_column_ranges(item_preprocessor):
    data = item_preprocessor.select_features(columns=":1,3:")
    assert "name" in data.columns
    assert "price" in data.columns


def test_select_features_with_regex(item_preprocessor):
    data = item_preprocessor.select_features(regex=".*e.*")
    assert "name" in data.columns
    assert "category" in data.columns
    assert "price" in data.columns


def test_simple_preprocessing_chain(item_preprocessor):
    # fmt: off
    data = (
        item_preprocessor
        .handle_missing_values(strategy="mean")
        .multilabel_binarize(["tags"])
        .normalize(["price"], methods=["z-score"])
        .onehot_encode(columns=["category"])
        .select_features(regex="^ftr.*", columns=["price"])
    )
    # fmt: on
    assert "price" in data.columns
    assert "ftr_T1" in data.columns
    assert "ftr_T2" in data.columns
    assert "ftr_T3" in data.columns
    assert "ftr_T4" in data.columns
    assert "ftr_T5" in data.columns
    assert "ftr_X" in data.columns
    assert "ftr_Y" in data.columns
    assert "ftr_Z" in data.columns
