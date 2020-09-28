"""Example using an sklearn Pipeline with early stopping.

Example taken and modified from
https://scikit-learn.org/stable/auto_examples/compose/
plot_compare_reduction.html
"""

from tune_sklearn import TuneSearchCV
from tune_sklearn import TuneGridSearchCV
from sklearn.datasets import load_digits
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

X, y = load_digits(return_X_y=True)

# partial_fit

pipe = Pipeline([("reduce_dim", PCA()), ("classify", SGDClassifier())])

param_grid = [
    {
        "classify__alpha": [1e-4, 1e-1, 1],
        "classify__epsilon": [0.01, 0.1]
    },
]

random = TuneSearchCV(
    pipe,
    param_grid,
    search_optimization="random",
    early_stopping=True,
    max_iters=10,
    pipeline_auto_early_stop=True)
random.fit(X, y)
print(random.cv_results_)

grid = TuneGridSearchCV(
    pipe,
    param_grid=param_grid,
    early_stopping=True,
    max_iters=10,
    pipeline_auto_early_stop=True)
grid.fit(X, y)
print(grid.cv_results_)

# warm start iter

pipe = Pipeline([("reduce_dim", PCA()), ("classify",
                                         LogisticRegression(max_iter=1000))])

param_grid = [
    {
        "classify__C": [1e-4, 1e-1, 1, 2]
    },
]

random = TuneSearchCV(
    pipe,
    param_grid,
    search_optimization="random",
    early_stopping=True,
    max_iters=10,
    pipeline_auto_early_stop=True)
random.fit(X, y)
print(random.cv_results_)

# warm start ensemble

pipe = Pipeline([("reduce_dim", PCA()), ("classify",
                                         RandomForestClassifier())])

param_grid = [
    {
        "classify__min_samples_split": [2, 3, 4]
    },
]

random = TuneSearchCV(
    pipe,
    param_grid,
    search_optimization="random",
    early_stopping=True,
    max_iters=10,
    pipeline_auto_early_stop=True)
random.fit(X, y)
print(random.cv_results_)
