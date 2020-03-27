"""
An example training a RandomForestClassifier, performing
randomized search using TuneRandomizedSearchCV. 
"""

from tune_sklearn.tune_search import TuneRandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.stats import randint
from ray.tune.schedulers import MedianStoppingRule
import numpy as np

digits = datasets.load_digits()
x = digits.data
y = digits.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

clf = RandomForestClassifier()
param_distributions = {
    'n_estimators': randint(20,80),
    'max_depth': randint(2,10)
}

tune_search = TuneRandomizedSearchCV(clf,
            param_distributions,
            n_iter=3,
            iters=10,
            )

tune_search.fit(x_train, y_train)

pred = tune_search.predict(x_test)
accuracy = np.count_nonzero(np.array(pred) == np.array(y_test))/len(pred)
print(accuracy)
