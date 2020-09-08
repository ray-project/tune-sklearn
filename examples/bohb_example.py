from tune_sklearn import TuneSearchCV

# Other imports
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

# Set training and validation sets
X, y = make_classification(
    n_samples=5000,
    n_features=200,
    n_informative=50,
    n_redundant=0,
    n_classes=10,
    class_sep=2.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000)

# Example parameter distributions to tune from SGDClassifier
# Note the use of tuples instead if non-random optimization is desired
param_dists = {"alpha": (1e-4, 1e-1), "epsilon": (1e-2, 1e-1)}

bohb_tune_search = TuneSearchCV(
    SGDClassifier(),
    param_distributions=param_dists,
    n_trials=20,
    max_iters=100,
    search_optimization="bohb",
)

bohb_tune_search.fit(X_train, y_train)
print(bohb_tune_search.best_params_)
