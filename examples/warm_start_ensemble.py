"""
An example training a LogisticRegression model, performing grid search
using TuneGridSearchCV.

This example uses early stopping to further improve runtimes
by eliminating worse hyperparameter choices early based off
of its average test score from cross validation. Usually
this will require the estimator to have `partial_fit`, but
we use sklearn's `warm_start` parameter to do this here.
We fit the estimator for one epoch, then `warm_start`
to pick up from where we left off, continuing until the
trial is early stopped or `max_iters` is reached.
"""

from tune_sklearn import TuneGridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

clf = RandomForestClassifier()
parameter_grid = {"min_samples_split": [2, 3, 4]}

tune_search = TuneGridSearchCV(
    clf,
    parameter_grid,
    early_stopping=True,
    max_iters=20,
)
tune_search.fit(x_train, y_train)

pred = tune_search.predict(x_test)
accuracy = np.count_nonzero(np.array(pred) == np.array(y_test)) / len(pred)
print(accuracy)
