import io
import os
from setuptools import setup, find_packages
from tune_sklearn import __version__

ROOT_DIR = os.path.dirname(__file__)

setup(
    name="tune_sklearn",
    packages=find_packages(),
    version=__version__,
    author="Michael Chau, Anthony Yu, and Ray Team",
    description=(
        "A drop-in replacement for Scikit-Learn’s "
        "GridSearchCV / RandomizedSearchCV with cutting edge "
        "hyperparameter tuning techniques."),
    long_description=io.open(
        os.path.join(ROOT_DIR, "README.md"),
        "r",
        encoding="utf-8").read(),
    url="https://github.com/ray-project/tune-sklearn",
    install_requires=["scikit-learn", "scipy", "ray[tune]", "numpy>=1.16"])
