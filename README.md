# tune-sklearn
[![Build Status](https://travis-ci.com/ray-project/tune-sklearn.svg?branch=master)](https://travis-ci.com/ray-project/tune-sklearn)

Tune-sklearn is a package that integrates Ray Tune's hyperparameter tuning and scikit-learn's models, allowing users to optimize hyerparameter searching for sklearn using Tune's schedulers. Tune-sklearn follows the same API as scikit-learn's GridSearchCV, but allows for more flexibility in defining hyperparameter search regions, such as distributions to sample from.

## Quick Start
`TuneGridSearchCV` example. The dataset used in the example (MNIST) can be found [here](https://drive.google.com/file/d/1XUkN4a6NcvB9Naq9Gy8wVlqfTKHqAVd5/view?usp=sharing). We use this dataset to exemplify the speedup factor of `TuneGridSearchCV`.
```python
from tune_sklearn.tune_search import TuneGridSearchCV

# Load in data
from scipy import io
data = io.loadmat("mnist_data.mat")

# Other imports
from sklearn.model_selection import train_test_split
from ray.tune.schedulers import MedianStoppingRule
from sklearn.linear_model import SGDClassifier

# Set training and validation sets
X = data["training_data"]
y = data["training_labels"].ravel()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=0)

# Example parameters to tune from SGDClassifier
parameters = {
    'alpha': [1e-4, 1e-1, 1],
    'epsilon':[0.01, 0.1]
}

size = 20000 # To save time
X_subset = X_train[:size]
y_subset = y_train[:size]

scheduler = MedianStoppingRule(metric='average_test_score',
                        grace_period=10.0)
tune_search = TuneGridSearchCV(SGDClassifier(),
                               parameters,
                               early_stopping=True,
                               scheduler=scheduler,
                               iters=10)
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
sklearn_search = GridSearchCV(SGDClassifier(), parameters, n_jobs=-1)

start = time.time()
sklearn_search.fit(X_subset, y_subset)
end = time.time()
print(“Sklearn Fit Time:”, end - start)
```


Or, you could also try the randomized search interface `TuneRandomizedSearchCV`:
```python
from tune_sklearn.tune_search import TuneRandomizedSearchCV
import scipy
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

clf = SGDClassifier()
param_grid = {
    'alpha': scipy.stats.uniform(1e-4, 1e-1)
}

tune_search = TuneRandomizedSearchCV(clf,
            param_distributions=param_grid,
            refit=True,
            early_stopping=True,
            iters=10)
tune_search.fit(x_train, y_train)

pred = tune_search.predict(x_test)
```

## More information
[Ray Tune](https://ray.readthedocs.io/en/latest/tune.html)
