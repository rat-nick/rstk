from setuptools import find_packages, setup

setup(
    name="rstk",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "click",
        "pandas",
        "numpy",
        "scikit-learn",
    ],
    entry_points="""
        [console_scripts]
        rstk=src.cli:cli
    """,
)
