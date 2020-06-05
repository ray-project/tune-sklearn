from setuptools import setup

setup(
    name="tune_sklearn",
    version="0.0.1",
    packages=["tune_sklearn"],
    author="Michael Chau/Anthony Yu",
    description="An experimental scikit-learn API on Tune",
    long_description="An API enabling faster scikit-learn training using Tune "
    "parallelization and early stopping algorithms",
    url="https://github.com/ray-project/tune-sklearn",
    install_requires=[
        "scikit-learn>=0.22", "scipy", "ray", "numpy>=1.16", "pandas",
        "tabulate", "parameterized"
    ])
