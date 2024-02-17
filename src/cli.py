import click
import pandas as pd

from .algo.content_based.knn import KNN


@click.group()
def main():
    pass


@main.command(
    help="Builds a recommendation model using the given algorithm on the given dataset"
)
@click.argument(
    "algorithm",
    type=click.Choice(
        ["content-knn", "user-cf-knn", "item-cf-knn"],
    ),
)
@click.argument("dataset", type=click.Path(exists=True))
@click.option(
    "--model-path",
    default="model.pkl",
    help="Path to the file where the model will be serialized",
    type=click.Path(),
)
def build(algorithm, dataset, model_path):
    click.echo("Building recommender system...")
    click.echo("Algorithm: %s" % algorithm)
    click.echo("Dataset: %s" % dataset)
    click.echo("Model path : %s" % model_path)

    df = pd.read_csv(dataset)

    if algorithm == "content-knn":
        model = KNN(df)
        model.serialize(model_path)
        print("Finished building content based KNN model.")


if __name__ == "__main__":
    main()
