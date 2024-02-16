from setuptools import setup, find_packages

setup(
    name='rstk',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'click',
        'pandas',
        'numpy',
        'scikit-learn',

    ],
    entry_points='''
        [console_scripts]
        rstk=rstk.cli:cli
    '''
)