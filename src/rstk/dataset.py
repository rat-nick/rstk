import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, data: pd.DataFrame | np.array) -> None:
        self.data = data
        self.inner2raw = {}
        self.raw2inner = {}
        self._build_translation_dicts()

    def _build_translation_dicts(self):
        self.inner2raw = {k: v for k, v in enumerate(self.data.index)}
        self.raw2inner = {v: k for k, v in enumerate(self.data.index)}

    def train_test_split(self):
        train, test = np.split(self.data, [int(0.8 * len(self.data))])
        return Trainset(train), Trainset(test)


class Trainset:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data.reset_index()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]
