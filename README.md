# tune-sklearn
[![Build Status](https://travis-ci.com/ray-project/tune-sklearn.svg?branch=master)](https://travis-ci.com/ray-project/tune-sklearn)

Tune-sklearn is a package that integrates Ray Tune's hyperparameter tuning and scikit-learn's models, allowing users to optimize hyerparameter searching for sklearn using Tune's schedulers (more details in the [Tune Documentation](http://tune.io/)). 
Tune-sklearn follows the same API as scikit-learn's GridSearchCV, but allows for more flexibility in defining hyperparameter search regions, such as distributions to sample from.

Tune-sklearn provides additional benefits if specifying a scheduler **with an estimator that supports early stopping**. The list of estimators that can be supported from scikit-learn can be found in [scikit-learn's documentation at section 8.1.1.3](https://scikit-learn.org/stable/modules/computing.html#strategies-to-scale-computationally-bigger-data). 

If the estimator does not support `early_stopping` and it is set to `True` when declaring `TuneGridSearchCV`/`TuneRandomizedSearchCV`, an error will return saying the estimator does not support `partial_fit` due to no `early_stopping` available from the estimator. To safeguard against this, the current default for `early_stopping=False`, which simply runs the grid search cross-validation on Ray's parallel back-end and ignores the scheduler. We are currently experimenting with ways to support early stopping for estimators that do not directly expose an `early_stopping` interface in their estimators -- stay tuned!

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

# Example parameters to tune from SGDClassifier
parameters = {
    'alpha': [1e-4, 1e-1, 1],
    'epsilon':[0.01, 0.1]
}

size = 20000 # To save time
X_subset = X_train[:size]
y_subset = y_train[:size]

tune_search = TuneGridSearchCV(
    SGDClassifier(),
    parameters,
    scheduler=MedianStoppingRule(grace_period=10.0),
    early_stopping_max_epochs=10
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

#### TuneRandomizedSearchCV

```python
from tune_sklearn.tune_search import TuneRandomizedSearchCV

# Load in data
from scipy import io
data = io.loadmat("mnist_data.mat")

# Other imports
import scipy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from ray.tune.schedulers import MedianStoppingRule

# Set training and validation sets
X = data["training_data"]
y = data["training_labels"].ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Example parameter distributions to tune from SGDClassifier
param_dists = {
    'alpha': scipy.stats.uniform(1e-4, 1e-1),
    'epsilon': scipy.stats.uniform(1e-2, 1e-1)
}

size = 20000 # To save time
X_subset = X_train[:size]
y_subset = y_train[:size]

tune_search = TuneRandomizedSearchCV(SGDClassifier(),
    param_distributions=param_dists,
    n_iter=2,
    scheduler=MedianStoppingRule(grace_period=10.0),
    early_stopping_max_epochs=10
)

tune_search.fit(X_subset, y_subset)
```

## In Progress
We are currently finding better ways to parallelize the entire grid search cross-validation process. We do not see a significant speedup thus far when we are not able to early stop. We are also working to integrate more familiar interfaces to make it compatible with our grid search and randomized search interface. We will continue to add more examples in the [examples folder](https://github.com/ray-project/tune-sklearn/tree/master/examples) as we continue to add support for other interfaces!

## More information
[Ray Tune](https://ray.readthedocs.io/en/latest/tune.html)
