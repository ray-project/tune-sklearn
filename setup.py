import io
import os
from setuptools import setup, find_packages
from tune_sklearn import __version__

ROOT_DIR = os.path.dirname(__file__)

VERSION = os.environ.get("TSK_RELEASE_VERSION", __version__)

setup(
    name="tune_sklearn",
    packages=find_packages(),
    version=VERSION,
    author="Michael Chau, Anthony Yu, and Ray Team",
    author_email="ray-dev@googlegroups.com",
    description=("A drop-in replacement for Scikit-Learn’s "
                 "GridSearchCV / RandomizedSearchCV with cutting edge "
                 "hyperparameter tuning techniques."),
    long_description=io.open(
        os.path.join(ROOT_DIR, "README.md"), "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ray-project/tune-sklearn",
    install_requires=["scikit-learn", "scipy", "ray[tune]", "numpy>=1.16"])
