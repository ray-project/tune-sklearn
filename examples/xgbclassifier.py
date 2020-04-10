"""
An example training a XGBClassifier, performing
randomized search using TuneRandomizedSearchCV.
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
from tune_sklearn.tune_search import TuneRandomizedSearchCV
from sklearn import datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

digits = datasets.load_digits()
x = digits.data
y = digits.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

# A parameter grid for XGBoost
params = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}
xgb = XGBClassifier(learning_rate=0.02, n_estimators=50, objective='binary:logistic',
                    silent=True, nthread=1)

from tune_sklearn.tune_search import TuneRandomizedSearchCV, TuneGridSearchCV
digit_search = TuneGridSearchCV(
    xgb, 
    param_grid=params,
    # param_distributions=params, 
    # n_iter=3, 
)

import time
start = time.time()
digit_search.fit(x_train, y_train)
end = time.time()
print(end - start)
print(digit_search.best_params_)
print(digit_search.cv_results_)