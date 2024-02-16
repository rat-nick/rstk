import click
import argparse
@click.group()
def cli():
    pass

@cli.command()
def build():
    click.echo("Building recommender system...")


def preprocess():
    click.echo("Preprocessing data...")

if __name__ == "__main__":
    cli()