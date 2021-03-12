"""
An example training a LGBMClassifier, performing
randomized search using TuneSearchCV.
"""

from tune_sklearn import TuneSearchCV
from sklearn import datasets
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

digits = datasets.load_digits()
x = digits.data
y = digits.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

params = {
    "min_split_gain": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "reg_alpha": [0.5, 1, 1.5, 2, 5],
    "reg_lambda": [0.5, 1, 1.5, 2, 5],
    "max_depth": [3, 4, 5],
}

lgbm = LGBMClassifier(
    learning_rate=0.02,
    n_estimators=50,
)

digit_search = TuneSearchCV(
    lgbm,
    param_distributions=params,
    n_trials=3,
    early_stopping=True,
    # use_gpu=True # Commented out for testing on github actions,
    # but this is how you would use gpu
)

digit_search.fit(x_train, y_train)
print(digit_search.best_params_)
print(digit_search.cv_results_)
