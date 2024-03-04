import pytest

from ..src.rstk.server import serve


@pytest.fixture()
def app():

    app = serve(None, port=0)
    app.config.update(
        {
            "TESTING": True,
        }
    )

    yield app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def runner(app):
    return app.test_cli_runner()
