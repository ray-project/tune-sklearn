"""
An example training a XGBClassifier, performing
randomized search using TuneSearchCV.
"""

from tune_sklearn import TuneSearchCV
from sklearn import datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

digits = datasets.load_digits()
x = digits.data
y = digits.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

# A parameter grid for XGBoost
params = {
    "min_child_weight": [1, 5, 10],
    "gamma": [0.5, 1, 1.5, 2, 5],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "max_depth": [3, 4, 5],
}

xgb = XGBClassifier(
    learning_rate=0.02,
    n_estimators=50,
    objective="binary:logistic",
    nthread=4,
    # tree_method="gpu_hist"  # this enables GPU.
    # See https://github.com/dmlc/xgboost/issues/2819
)

digit_search = TuneSearchCV(
    xgb,
    param_distributions=params,
    n_trials=3,
    early_stopping=True,
    # use_gpu=True # Commented out for testing on github actions,
    # but this is how you would use gpu
)

digit_search.fit(x_train, y_train)
print(digit_search.best_params_)
print(digit_search.cv_results_)
