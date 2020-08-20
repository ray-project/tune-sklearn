"""
An example training a RandomForestClassifier, performing
randomized search using TuneSearchCV.
"""

from tune_sklearn import TuneSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import time

digits = datasets.load_digits()
x = digits.data
y = digits.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

clf = RandomForestClassifier()
param_distributions = {
    "n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "criterion": ["gini", "entropy"],
    "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
    "min_samples_split": [2, 5, 7, 9, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["auto", "sqrt", "log2"],
    "bootstrap": [True, False]
}

tune_search = TuneSearchCV(clf, param_distributions, n_jobs=1, n_iter=15)
start = time.time()
tune_search.fit(x_train, y_train)
end = time.time()
tune_fit_time = end - start
print("Tune Fit Time:", tune_fit_time)
pred = tune_search.predict(x_test)
accuracy = np.count_nonzero(np.array(pred) == np.array(y_test)) / len(pred)
print(accuracy)

random_search = RandomizedSearchCV(clf, param_distributions, n_iter=15)
start = time.time()
random_search.fit(x_train, y_train)
end = time.time()
random_fit_time = end - start
print("Random Fit Time:", random_fit_time)
pred = random_search.predict(x_test)
accuracy = np.count_nonzero(np.array(pred) == np.array(y_test)) / len(pred)
print(accuracy)

assert tune_fit_time > random_fit_time, (
    "Tune has regressed in comparison to random search?")
