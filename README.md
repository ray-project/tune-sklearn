# tune-sklearn
[![Build Status](https://travis-ci.com/ray-project/tune-sklearn.svg?branch=master)](https://travis-ci.com/ray-project/tune-sklearn)

Tune-sklearn is a package that integrates Ray Tune's hyperparameter tuning and scikit-learn's models, allowing users to optimize hyerparameter searching for sklearn using Tune's schedulers (more details in the [Tune Documentation](http://tune.io/)). 
Tune-sklearn follows the same API as scikit-learn's GridSearchCV, but allows for more flexibility in defining hyperparameter search regions, such as distributions to sample from.

Tune-sklearn provides additional benefits if specifying a scheduler **with an estimator that supports early stopping**. The list of estimators that can be supported from scikit-learn can be found in [scikit-learn's documentation at section 8.1.1.3](https://scikit-learn.org/stable/modules/computing.html#strategies-to-scale-computationally-bigger-data). 

If the estimator does not support `partial_fit`, a warning will be shown saying early stopping cannot be done and it will simply run the cross-validation on Ray's parallel back-end. We are currently experimenting with ways to support early stopping for estimators that do not directly expose a `partial_fit` interface in their estimators -- stay tuned! By default, early stopping is performed whenever possible, but can be disabled by setting the `early_stopping` parameter to `false`.

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
    scheduler="MedianStoppingRule",
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

#### TuneSearchCV

`TuneSearchCV` uses randomized search over the distribution by default, but can do Bayesian search as well by specifying the `search_optimization` parameter as shown here. You need to run `pip install bayesian-optimization` for this to work. More details in [Tune Documentation](https://docs.ray.io/en/latest/tune-searchalg.html#bayesopt-search).

```python
from tune_sklearn.tune_search import TuneSearchCV

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
    'alpha': (1e-4, 1e-1),
    'epsilon': (1e-2, 1e-1)
}

size = 20000 # To save time
X_subset = X_train[:size]
y_subset = y_train[:size]

tune_search = TuneSearchCV(SGDClassifier(),
    param_distributions=param_dists,
    n_iter=2,
    scheduler="MedianStoppingRule",
    early_stopping_max_epochs=10,
    search_optimization="bayesian"
)

tune_search.fit(X_subset, y_subset)
```

### Other Machine Learning Libraries

#### Pytorch/skorch

```python
import numpy as np
from sklearn.datasets import make_classification
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from tune_sklearn.tune_search import TuneGridSearchCV

X, y = make_classification(1000, 20, n_informative=10, random_state=0)
X = X.astype(np.float32)
y = y.astype(np.int64)


class MyModule(nn.Module):
    def __init__(self, num_units=10, nonlin=F.relu):
        super(MyModule, self).__init__()
        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X))
        return X


net = NeuralNetClassifier(
    MyModule,
    max_epochs=10,
    lr=0.1,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
)

params = {
    "lr": [0.01, 0.02],
    "module__num_units": [10, 20],
}

gs = TuneGridSearchCV(net, params, scoring="accuracy")
gs.fit(X, y)
print(gs.best_score_, gs.best_params_)
```

#### Keras

```python
from keras.datasets import mnist
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from tune_sklearn.tune_search import TuneGridSearchCV

nb_classes = 10
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train[:500]
y_train = y_train[:500]
X_test = X_test[:100]
y_test = y_test[:100]

X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


def create_model(optimizer="rmsprop", init="glorot_uniform"):
    model = Sequential()
    model.add(Dense(512, input_shape=(784, )))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(512, init=init))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(10, init=init))
    model.add(Activation("softmax"))  # This special "softmax" a
    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


model = KerasClassifier(build_fn=create_model)
optimizers = ["rmsprop", "adam"]
init = ["glorot_uniform", "normal"]
epochs = [5, 10]
param_grid = dict(optimizer=optimizers, nb_epoch=epochs, init=init)
grid = TuneGridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_train, Y_train)
print(grid_result.best_params_)
print(grid_result.cv_results_)
```

Check our examples folder for other examples of use cases.

## In Progress
We are currently finding better ways to parallelize the entire grid search cross-validation process. We do not see a significant speedup thus far when we are not able to early stop. We are also working to integrate more familiar interfaces to make it compatible with our grid search and randomized search interface. We will continue to add more examples in the [examples folder](https://github.com/ray-project/tune-sklearn/tree/master/examples) as we continue to add support for other interfaces!

## More information
[Ray Tune](https://ray.readthedocs.io/en/latest/tune.html)
