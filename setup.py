from distutils.core import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='tune_sklearn',
    version='0.1',
    packages=['tune_sklearn'],
    author='Michael Chau/Anthony Yu',
    description='An experimental scikit-learn API on Tune',
    long_description=long_description,
    url='https://github.com/ray-project/tune-sklearn',
    install_requires=['scikit-learn>=0.22', 'scipy', 'ray', 'numpy>=1.16', 'pandas', 'tabulate', 'parameterized']
)
