# tune-sklearn
[![Build Status](https://travis-ci.com/ray-project/tune-sklearn.svg?branch=master)](https://travis-ci.com/ray-project/tune-sklearn)

Tune-sklearn is a drop-in replacement for Scikit-Learn’s model selection module (GridSearchCV, RandomizedSearchCV) with cutting edge hyperparameter tuning techniques.

## Features
Here’s what tune-sklearn has to offer:

 * **Consistency with Scikit-Learn API**: Change less than 5 lines in a standard Scikit-Learn script to use the API [[example](https://github.com/ray-project/tune-sklearn/blob/master/examples/random_forest.py)].
 * **Modern tuning techniques**: tune-sklearn allows you to easily leverage Bayesian Optimization, HyperBand, BOHB, and other optimization techniques by simply toggling a few parameters.
 * **Framework support**: tune-sklearn is used primarily for tuning Scikit-Learn models, but it also supports and provides examples for many other frameworks with Scikit-Learn wrappers such as Skorch (Pytorch) [[example](https://github.com/ray-project/tune-sklearn/blob/master/examples/torch_nn.py)], KerasClassifier (Keras) [[example](https://github.com/ray-project/tune-sklearn/blob/master/examples/keras_example.py)], and XGBoostClassifier (XGBoost) [[example](https://github.com/ray-project/tune-sklearn/blob/master/examples/xgbclassifier.py)].
 * **Scale up**: Tune-sklearn leverages [Ray Tune](http://tune.io/), a library for distributed hyperparameter tuning, to parallelize cross validation on multiple cores and even multiple machines without changing your code.

Check out our [API Documentation](https://docs.ray.io/en/master/tune/api_docs/sklearn.html) and [Walkthrough](https://docs.ray.io/en/master/tune/tutorials/tune-sklearn.html) (for `master` branch).

## Installation

### Dependencies
- numpy (>=1.16)
- [ray](http://docs.ray.io/)
- scikit-learn (>=0.23)

### User Installation

`pip install tune-sklearn ray[tune]`

or

`pip install -U git+https://github.com/ray-project/tune-sklearn.git && pip install 'ray[tune]'`


### Tune-sklearn Early Stopping

For certain estimators, tune-sklearn can also immediately enable **incremental training and early stopping**. Such estimators include:
 * Estimators that implement 'warm_start' (except for ensemble classifiers and decision trees)
 * Estimators that implement partial fit
 * [XGBoost](https://github.com/dmlc/xgboost/issues/1686), LightGBM and [CatBoost](https://catboost.ai/docs/concepts/python-reference_train.html?lang=en) models (via incremental learning)

To read more about compatible scikit-learn models, see [scikit-learn's documentation at section 8.1.1.3](https://scikit-learn.org/stable/modules/computing.html#strategies-to-scale-computationally-bigger-data).

Early stopping algorithms that can be enabled include HyperBand and Median Stopping (see below for examples).

If the estimator does not support `partial_fit`, a warning will be shown saying early stopping cannot be done and it will simply run the cross-validation on Ray's parallel back-end.

Apart from early stopping scheduling algorithms, tune-sklearn also supports passing custom stoppers to Ray Tune. These
can be passed via the `stopper` argument when instantiating `TuneSearchCV` or `TuneGridSearchCV`.
See [the Ray documentation for an overview of available stoppers](https://docs.ray.io/en/master/tune/api_docs/stoppers.html).

## Examples

#### TuneGridSearchCV
To start out, it’s as easy as changing our import statement to get Tune’s grid search cross validation interface, and the rest is almost identical!

`TuneGridSearchCV` accepts dictionaries in the format `{ param_name: str : distribution: list }` or a list of such dictionaries, just like scikit-learn's `GridSearchCV`. The distribution can also be the output of Ray Tune's [`tune.grid_search`](https://docs.ray.io/en/master/tune/api_docs/search_space.html#grid-search-api).

```python
# from sklearn.model_selection import GridSearchCV
from tune_sklearn import TuneGridSearchCV

# Other imports
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

# Set training and validation sets
X, y = make_classification(n_samples=11000, n_features=1000, n_informative=50, n_redundant=0, n_classes=10, class_sep=2.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000)

# Example parameters to tune from SGDClassifier
parameters = {
    'alpha': [1e-4, 1e-1, 1],
    'epsilon':[0.01, 0.1]
}

tune_search = TuneGridSearchCV(
    SGDClassifier(),
    parameters,
    early_stopping="MedianStoppingRule",
    max_iters=10
)

import time # Just to compare fit times
start = time.time()
tune_search.fit(X_train, y_train)
end = time.time()
print("Tune Fit Time:", end - start)
pred = tune_search.predict(X_test)
accuracy = np.count_nonzero(np.array(pred) == np.array(y_test)) / len(pred)
print("Tune Accuracy:", accuracy)
```

If you'd like to compare fit times with sklearn's `GridSearchCV`, run the following block of code:

```python
from sklearn.model_selection import GridSearchCV
# n_jobs=-1 enables use of all cores like Tune does
sklearn_search = GridSearchCV(
    SGDClassifier(),
    parameters,
    n_jobs=-1
)

start = time.time()
sklearn_search.fit(X_train, y_train)
end = time.time()
print("Sklearn Fit Time:", end - start)
pred = sklearn_search.predict(X_test)
accuracy = np.count_nonzero(np.array(pred) == np.array(y_test)) / len(pred)
print("Sklearn Accuracy:", accuracy)
```

#### TuneSearchCV

`TuneSearchCV` is an upgraded version of scikit-learn's `RandomizedSearchCV`.

It also provides a wrapper for several search optimization algorithms from Ray Tune's [`tune.suggest`](https://docs.ray.io/en/master/tune/api_docs/suggestion.html), which in turn are wrappers for other libraries. The selection of the search algorithm is controlled by the `search_optimization` parameter. In order to use other algorithms, you need to install the libraries they depend on (`pip install` column). The search algorithms are as follows:

| Algorithm          | `search_optimization` value | Summary                | Website                                                 | `pip install`              |
|--------------------|-----------------------------|------------------------|---------------------------------------------------------|--------------------------|
| (Random Search)    | `"random"`                  | Randomized Search      |                                                         | built-in                 |
| SkoptSearch        | `"bayesian"`                | Bayesian Optimization  | [[Scikit-Optimize](https://scikit-optimize.github.io/)] | `scikit-optimize`        |
| HyperOptSearch     | `"hyperopt"`                | Tree-Parzen Estimators | [[HyperOpt](http://hyperopt.github.io/hyperopt)]        | `hyperopt`               |
| TuneBOHB           | `"bohb"`                    | Bayesian Opt/HyperBand | [[BOHB](https://github.com/automl/HpBandSter)]          | `hpbandster ConfigSpace` |
| Optuna             | `"optuna"`                  | Tree-Parzen Estimators | [[Optuna](https://optuna.readthedocs.io/en/stable/)]    | `optuna`                 |

All algorithms other than RandomListSearcher accept parameter distributions in the form of dictionaries in the format `{ param_name: str : distribution: tuple or list }`.

Tuples represent real distributions and should be two-element or three-element, in the format `(lower_bound: float, upper_bound: float, Optional: "uniform" (default) or "log-uniform")`. Lists represent categorical distributions. [Ray Tune Search Spaces](https://docs.ray.io/en/master/tune/api_docs/search_space.html) are also supported. Furthermore, each algorithm also accepts parameters in their own specific format. More information in [Tune documentation](https://docs.ray.io/en/master/tune/api_docs/suggestion.html).

Random Search (default) accepts dictionaries in the format `{ param_name: str : distribution: list }` or a list of such dictionaries, just like scikit-learn's `RandomizedSearchCV`.

```python
from tune_sklearn import TuneSearchCV

# Other imports
import scipy
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

# Set training and validation sets
X, y = make_classification(n_samples=11000, n_features=1000, n_informative=50, n_redundant=0, n_classes=10, class_sep=2.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000)

# Example parameter distributions to tune from SGDClassifier
# Note the use of tuples instead if non-random optimization is desired
param_dists = {
    'alpha': (1e-4, 1e-1),
    'epsilon': (1e-2, 1e-1)
}

bohb_tune_search = TuneSearchCV(SGDClassifier(),
    param_distributions=param_dists,
    n_trials=2,
    max_iters=10,
    search_optimization="bohb"
)

bohb_tune_search.fit(X_train, y_train)

hyperopt_tune_search = TuneSearchCV(SGDClassifier(),
    param_distributions=param_dists,
    n_trials=2,
    early_stopping=True, # uses Async HyperBand if set to True
    max_iters=10,
    search_optimization="hyperopt"
)

hyperopt_tune_search.fit(X_train, y_train)
```

### Other Machine Learning Libraries and Examples
Tune-sklearn also supports the use of other machine learning libraries such as Pytorch (using Skorch) and Keras. You can find these examples here:
* [Keras](https://github.com/ray-project/tune-sklearn/blob/master/examples/keras_example.py)
* [LightGBM](https://github.com/ray-project/tune-sklearn/blob/master/examples/lgbm.py)
* [Sklearn Random Forest](https://github.com/ray-project/tune-sklearn/blob/master/examples/random_forest.py)
* [Sklearn Pipeline](https://github.com/ray-project/tune-sklearn/blob/master/examples/sklearn_pipeline.py)
* [Pytorch (Skorch)](https://github.com/ray-project/tune-sklearn/blob/master/examples/torch_nn.py)
* [XGBoost](https://github.com/ray-project/tune-sklearn/blob/master/examples/xgbclassifier.py)

## More information
[Ray Tune](https://docs.ray.io/en/latest/tune/index.html)
