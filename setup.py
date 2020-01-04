from distutils.core import setup

setup(
    name='tune-sklearn',
    version='0.1',
    packages=['tune_sklearn'],
    author='Michael Chau/Anthony Yu',
    install_requires=['scikit-learn', 'scipy', 'ray', 'numpy']
)
