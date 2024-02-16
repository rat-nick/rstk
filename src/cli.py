import click


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--algorithm",
    "-a",
    required=True,
    type=click.Choice(["content-knn", "colaborative-knn"]),
)
@click.option("--data", "-d", required=True)
def build(algorithm, data):
    click.echo("Building recommender system...")
    click.echo("Algorithm: %s" % algorithm)
    click.echo("Dataset: %s" % data)


def preprocess():
    click.echo("Preprocessing data...")


if __name__ == "__main__":
    cli()
