import io
import os
from setuptools import setup, find_packages

ROOT_DIR = os.path.dirname(__file__)

# Ray tests depend on buidling tune_sklearn. Thus, if `setup.py`
# depends on Ray, then we create a cyclic dep.
# workaround from: https://stackoverflow.com/a/17626524
with open("tune_sklearn/_version.py") as f:
    text = f.readlines()  # Returns ['__version__ = "0.2.0"']
    __version__ = text[-1].split()[-1].strip("\"'")

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
