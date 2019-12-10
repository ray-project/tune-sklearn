from tune_sklearn import TuneRandomizedSearchCV, TuneGridSearchCV
from scipy.stats import randint, uniform
from sklearn.model_selection import GridSearchCV
from ray import tune
import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
import scipy.sparse as sp
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.datasets import (
    make_blobs,
    make_classification,
    make_multilabel_classification,
)
from sklearn.metrics import accuracy_score, f1_score, make_scorer, roc_auc_score
from sklearn.base import BaseEstimator
import pytest
from sklearn.cluster import KMeans
from sklearn.svm import SVC, LinearSVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KernelDensity
from ray.tune.schedulers import MedianStoppingRule
import unittest
from test_utils import (
    MockClassifier,
    CheckingClassifier,
    BrokenClassifier,
    MockDataFrame

)
# test grid-search with an estimator without predict.
# slight duplication of a test from KDE
def custom_scoring(estimator, X):
    return 42 if estimator.bandwidth == 0.1 else 0

X, _ = make_blobs(cluster_std=0.1, random_state=1, centers=[[0, 1], [1, 0], [0, 0]])
search = TuneGridSearchCV(
    KernelDensity(),
    param_grid=dict(bandwidth=[0.01, 0.1, 1]),
    scoring=custom_scoring,
)
search.fit(X)
print(search.best_score_)
