# tune-sklearn
[![Build Status](https://travis-ci.com/ray-project/tune-sklearn.svg?branch=master)](https://travis-ci.com/ray-project/tune-sklearn)

Tune-sklearn is a package that integrates Ray Tune's hyperparameter tuning and scikit-learn's models, allowing users to optimize hyerparameter searching for sklearn using Tune's schedulers. Tune-sklearn follows the same API as scikit-learn's GridSearchCV, but allows for more flexibility in defining hyperparameter search regions, such as distributions to sample from.

## Quick Start
Use tune-sklearn TuneGridSearchCV to tune sklearn model
```python
from tune_sklearn.tune_search import TuneGridSearchCV
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from ray.tune.schedulers import MedianStoppingRule

# Loading the Digits dataset
digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = {'kernel': ['rbf'],
                    'gamma': [1e-3, 1e-4],
                    'C': [1, 10, 100, 1000]
                    }

tune_search = TuneGridSearchCV(SVC(),
                               tuned_parameters,
                               scheduler=MedianStoppingRule())
tune_search.fit(X_train, y_train)

pred = tune_search.predict(X_test)

correct = 0
for i in range(len(pred)):
    if pred[i] == y_test[i]:
        correct += 1
print(correct / len(pred))
print(tune_search.cv_results_)
```

Use tune-sklearn TuneRandomizedSearchCV to tune sklearn model

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
            n_jobs=5,
            refit=True,
            early_stopping=True,
            iters=10)
tune_search.fit(x_train, y_train)

pred = tune_search.predict(x_test)
```

## More information
[Ray Tune](https://ray.readthedocs.io/en/latest/tune.html)
