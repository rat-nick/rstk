from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="rstk",
    version="0.1.1",
    package_dir={"rstk": "src"},
    install_requires=[
        "click",
        "pandas",
        "numpy",
        "scikit-learn",
    ],
    entry_points="""
        [console_scripts]
        rstk=rstk.cli:main
    """,
    author="Nikola Ratinac",
    author_email="nikola.ratinac@gmail.com",
    description="Lightweight recommender system toolkit",
    long_description=long_description,
    url="https://github.com/rat-nick/rstk",
)
