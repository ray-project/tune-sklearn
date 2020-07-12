# tune-sklearn
[![Build Status](https://travis-ci.com/ray-project/tune-sklearn.svg?branch=master)](https://travis-ci.com/ray-project/tune-sklearn)

Tune-sklearn is a package that integrates Ray Tune's hyperparameter tuning and scikit-learn's models, allowing users to optimize hyerparameter searching for sklearn using Tune's schedulers (more details in the [Tune Documentation](http://tune.io/)). 
Tune-sklearn follows the same API as scikit-learn's GridSearchCV, but allows for more flexibility in defining hyperparameter search regions, such as distributions to sample from.

Tune-sklearn provides additional benefits if specifying a scheduler **with an estimator that supports early stopping**. The list of estimators that can be supported from scikit-learn can be found in [scikit-learn's documentation at section 8.1.1.3](https://scikit-learn.org/stable/modules/computing.html#strategies-to-scale-computationally-bigger-data). 

If the estimator does not support `partial_fit`, a warning will be shown saying early stopping cannot be done and it will simply run the cross-validation on Ray's parallel back-end.

### Installation

#### Dependencies
- numpy (>=1.16)
- ray
- scikit-learn (>=0.23)

#### User Installation

`pip install tune-sklearn ray[tune]`

## Examples
#### TuneGridSearchCV
To start out, it’s as easy as changing our import statement to get Tune’s grid search cross validation interface, and the rest is almost identical!

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

`TuneSearchCV` uses randomized search over the distribution by default, but can do Bayesian search as well by specifying the `search_optimization` parameter as shown here. You need to run `pip install scikit-optimize` for this to work. More details in [Tune Documentation] (https://docs.ray.io/en/master/tune/api_docs/suggestion.html?highlight=bayesian#scikit-optimize-tune-suggest-skopt-skoptsearch).

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
# Note the use of tuples instead if Bayesian optimization is desired
param_dists = {
    'alpha': (1e-4, 1e-1),
    'epsilon': (1e-2, 1e-1)
}

tune_search = TuneSearchCV(SGDClassifier(),
    param_distributions=param_dists,
    n_iter=2,
    early_stopping="MedianStoppingRule",
    max_iters=10,
    search_optimization="bayesian"
)

tune_search.fit(X_train, y_train)
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
[Ray Tune](https://ray.readthedocs.io/en/latest/tune.html)
