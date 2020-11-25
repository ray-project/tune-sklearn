from setuptools import setup

setup(
    name="tune_sklearn",
    packages=["tune_sklearn"],
    version="0.0.8",
    author="Michael Chau/Anthony Yu",
    description="An experimental scikit-learn API on Tune",
    long_description="An API enabling faster scikit-learn training using Tune "
    "parallelization and early stopping algorithms",
    url="https://github.com/ray-project/tune-sklearn",
    install_requires=["scikit-learn", "scipy", "ray[tune]", "numpy>=1.16"])
