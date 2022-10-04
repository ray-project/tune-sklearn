"""
An example using a Searcher (HEBOSearch) not defined
directly in TuneSearchCV.
"""

from tune_sklearn import TuneSearchCV
from ray import tune
from ray.tune.search.hebo import HEBOSearch
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

digits = datasets.load_digits()
x = digits.data
y = digits.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

clf = RandomForestClassifier()
param_distributions = {
    "n_estimators": tune.randint(20, 80),
    "max_depth": tune.randint(2, 10)
}

# If a Searcher is initialized without specifying search space,
# it will use the space passed to TuneSearchCV
# (in this example: param_distributions)
searcher = HEBOSearch()

# It is also possible to use user-defined Searchers, as long as
# they inherit from ray.tune.search.Searcher and have the following
# attributes: _space, _metric, _mode

tune_search = TuneSearchCV(
    clf, param_distributions, n_trials=3, search_optimization=searcher)

tune_search.fit(x_train, y_train)

pred = tune_search.predict(x_test)
accuracy = np.count_nonzero(np.array(pred) == np.array(y_test)) / len(pred)
print(accuracy)
