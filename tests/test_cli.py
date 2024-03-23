import os
import subprocess

import requests
from click.testing import CliRunner

from ..src.rstk import cli


def test_rstk_build():
    runner = CliRunner()
    result = runner.invoke(
        cli.build,
        [
            "content-based-similarity",
            "data/dataset.csv",
            "--delimiter",
            "|",
            "--model-path",
            "models/knn.pkl",
            "--id-column",
            "movie id",
        ],
    )

    assert result.exit_code == 0
    assert "Building recommender system..." in result.output
    assert "Algorithm: content-based-similarity" in result.output
    assert "Dataset: data/dataset.csv" in result.output
    assert "Model path: models/knn.pkl" in result.output
    assert os.path.exists("models/knn.pkl")
