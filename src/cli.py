import click


@click.group()
def cli():
    pass


@cli.command(
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


def preprocess():
    click.echo("Preprocessing data...")


if __name__ == "__main__":
    cli()
