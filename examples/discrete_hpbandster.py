from tune_sklearn import TuneSearchCV
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import ConfigSpace as CS

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

space = {
    "n_estimators": CS.UniformIntegerHyperparameter("n_estimators", 100, 200),
    "min_weight_fraction_leaf": (0.0, 0.5),
    "min_samples_leaf": CS.UniformIntegerHyperparameter(
        "min_samples_leaf", 1, 5)
}

tune_search = TuneSearchCV(
    SGDClassifier(),
    space,
    search_optimization="bohb",
    n_trials=3,
    max_iters=10)
tune_search.fit(X_train, y_train)

print(tune_search.cv_results_)
print(tune_search.best_params_)
