# tune-sklearn
[![Build Status](https://travis-ci.com/ray-project/tune-sklearn.svg?branch=master)](https://travis-ci.com/ray-project/tune-sklearn)

Tune-sklearn is a package that integrates Ray Tune's hyperparameter tuning and scikit-learn's models, allowing users to optimize hyerparameter searching for sklearn using Tune's schedulers (more details in the [Tune Documentation](http://tune.io/)).

Tune-sklearn follows the same API as scikit-learn's GridSearchCV, but with Tune's back-end, there is massive speedup of up to 10 times for large datasets.

Tune-sklearn provides additional benefits if specifying a scheduler **with an estimator that supports early stopping**. The list of estimators that can be supported from scikit-learn can be found in [scikit-learn's documentation at section 8.1.1.3](https://scikit-learn.org/stable/modules/computing.html#strategies-to-scale-computationally-bigger-data).

Early stopping can be activated by setting `early_stopping=True` in `TuneGridSearchCV`/`TuneSearchCV` (default is `False`). If the estimator does not support `partial_fit`, a warning will be shown saying early stopping cannot be done and it will simply run the cross-validation on Ray's parallel back-end.
### Installation

#### Dependencies
- numpy (>=1.16)
- ray
- scikit-learn (>=0.22)
- cloudpickle

#### User Installation

`pip install tune-sklearn`

## Examples
#### TuneGridSearchCV
`TuneGridSearchCV` example. The dataset used in the example (MNIST) can be found [here](https://drive.google.com/file/d/1XUkN4a6NcvB9Naq9Gy8wVlqfTKHqAVd5/view?usp=sharing). We use this dataset to exemplify the speedup factor of `TuneGridSearchCV`.

```python
from tune_sklearn.tune_search import TuneGridSearchCV

# Load in data
from scipy import io
data = io.loadmat("mnist_data.mat")

# Other imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from ray.tune.schedulers import MedianStoppingRule

# Set training and validation sets
X = data["training_data"]
y = data["training_labels"].ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

size = 20000 # To save time
X_subset = X_train[:size]
y_subset = y_train[:size]

# Example parameters to tune from SGDClassifier
parameters = {
    'alpha': [1e-4, 1e-1, 1],
    'epsilon':[0.01, 0.1]
}

tune_search = TuneGridSearchCV(
    SGDClassifier(),
    parameters,
    scheduler="MedianStoppingRule",
    early_stopping=True,
    max_iters=10
)

import time # Just to compare fit times
start = time.time()
tune_search.fit(X_subset, y_subset)
end = time.time()
print(“Tune Fit Time:”, end - start)
```

If you'd like to compare fit times with sklearn's `GridSearchCV`, run the following block of code:

```python
# Use same X_subset, y_subset as above

from sklearn.model_selection import GridSearchCV
# n_jobs=-1 enables use of all cores like Tune does
sklearn_search = GridSearchCV(
    SGDClassifier(),
    parameters, 
    n_jobs=-1
)

start = time.time()
sklearn_search.fit(X_subset, y_subset)
end = time.time()
print(“Sklearn Fit Time:”, end - start)
```

#### TuneSearchCV

`TuneSearchCV` uses randomized search over the distribution by default, but can do Bayesian search as well by specifying the `search_optimization` parameter as shown here. You need to run `pip install bayesian-optimization` for this to work. More details in [Tune Documentation](https://docs.ray.io/en/latest/tune-searchalg.html#bayesopt-search).

Substitute the `parameters` and `tune_search` variables to use appropriate parameter distributions to invoke Bayesian optimization with `TuneSearchCV`.

```python
# Same setup as with GridSearchCV

# Example parameter distributions to tune from SGDClassifier
param_dists = {
    'alpha': (1e-4, 1e-1),
    'epsilon': (1e-2, 1e-1)
}

tune_search = TuneSearchCV(SGDClassifier(),
    param_distributions=param_dists,
    n_iter=2,
    scheduler="MedianStoppingRule",
    early_stopping=True,
    max_iters=10,
    search_optimization="bayesian"
)

tune_search.fit(X_subset, y_subset)
```

#### Other Machine Learning Libraries

We have also added support for other libraries such as PyTorch (with skorch) and Keras. Check out all the examples [here](https://github.com/ray-project/tune-sklearn/tree/master/examples).

## More information
[Ray Tune](https://ray.readthedocs.io/en/latest/tune.html)
