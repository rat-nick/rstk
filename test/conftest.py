import pytest
import pandas as pd
from src.preprocessor import Preprocessor


@pytest.fixture
def user_df() -> pd.DataFrame:
    data = {
        "userId": [0, 1, 2, 3, 4],
        "gender": ["M", "F", "M", "U", "F"],
        "age": [32, 19, 23, 21, 18],
    }
    return pd.DataFrame(data).set_index("userId")


@pytest.fixture
def user_preprocessor(user_df) -> Preprocessor:
    return Preprocessor(df=user_df)

@pytest.fixture
def item_df() -> pd.DataFrame:
    data = {
        "itemId": [0, 1, 2, 3, 4, 5, 6],
        "name": ["A", "B", "C", "D", "E", "F", "G"],
        "category" : ["X", "Y", "X", "X", None, "Z", "Z"],
        "tags" : ["T1|T2", None, "T2|T3", "T5|T2", "T5|T1|T4", "T3", "T4"],
        "price": [45, 34, 12, 80, 75, 25, None]
    }

    return pd.DataFrame(data).set_index("itemId")

@pytest.fixture
def item_preprocessor(item_df):
    return Preprocessor(df=item_df)