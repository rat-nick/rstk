"""
This module defines the CLI functionality for the RSTK library.
"""

import pickle

import click
import pandas as pd

from .data.dataset import FeatureDataset
from .engine import CBSEngine
from .model import SimilarityBased
from .preprocess import Preprocessor
from .server import serve


@click.group()
@click.version_option(version="0.2.0")
def main():
    """
    This is a click command group associated with the `rstk` command.
    """
    pass


@main.command(
    help="Builds a recommendation model using the given algorithm on the given dataset"
)
@click.argument(
    "algorithm",
    type=click.Choice(
        [
            "content-based-similarity",
            "colaborative-filtering-similarity",
        ],
    ),
)
@click.argument("dataset", type=click.Path(exists=True))
@click.option(
    "--model-path",
    default="model.pkl",
    help="Path to the file where the model will be serialized",
    type=click.Path(),
)
@click.option(
    "--delimiter",
    default=",",
    help="The delimiter used in the dataset",
    type=str,
)
@click.option(
    "--id-column",
    help="The name of the column in the dataset that contains the item IDs",
    type=str,
)
def build(algorithm, dataset, model_path, delimiter, id_column):
    """
    Builds a recommendation model using the given algorithm on the given dataset
    """
    click.echo("Building recommender system...")
    click.echo("Algorithm: %s" % algorithm)
    click.echo("Dataset: %s" % dataset)
    click.echo("Model path: %s" % model_path)

    if algorithm == "content-based-similarity":
        # load the data
        data = pd.read_csv(dataset, delimiter=delimiter, index_col=id_column)
        # perform preprocessing
        pp = Preprocessor(df=data)
        pp = pp.handle_missing_values()[:]
        # create the Dataset object
        dataset = FeatureDataset(data)
        engine = CBSEngine(dataset=dataset, model=SimilarityBased())
        click.echo("Training model...")
        engine.model.fit(dataset)
        click.echo("Saving model...")

        with open(model_path, "wb") as f:
            pickle.dump(engine, f)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.argument("port", type=int, default=11235)
def run(path, port):
    """
    A command-line function that serves a model on a specified port.
    It takes a file path and port number as arguments and loads the model from the file using pickle.
    """
    click.echo("Serving model...")
    click.echo("Model path : %s" % path)
    click.echo("Port : %s" % port)

    with open(path, "rb") as f:
        model = pickle.load(f)
        serve(model, port=port)


if __name__ == "__main__":
    main()
