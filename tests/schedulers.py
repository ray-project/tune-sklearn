"""
An example training an SGDClassifier, performing grid search
using TuneGridSearchCV.

This example uses early stopping to further improve runtimes
by eliminating worse hyperparameter choices early based off
of its average test score from cross validation.
"""

from tune_sklearn import TuneGridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--scheduler", type=str, required=True)
args = parser.parse_args()

digits = datasets.load_digits()
x = digits.data
y = digits.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

clf = SGDClassifier()
parameter_grid = {"alpha": [1e-4, 1e-1, 1], "epsilon": [0.01, 0.1]}

tune_search = TuneGridSearchCV(
    clf,
    parameter_grid,
    early_stopping=args.scheduler,
    max_iters=10,
)
tune_search.fit(x_train, y_train)

pred = tune_search.predict(x_test)
accuracy = np.count_nonzero(np.array(pred) == np.array(y_test)) / len(pred)
print(accuracy)
