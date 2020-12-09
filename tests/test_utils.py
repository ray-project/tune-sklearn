import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import _num_samples, check_array


class MockClassifier:
    """Dummy classifier to test the parameter search algorithms"""

    def __init__(self, foo_param=0, bar_param=0):
        self.count = 0
        self.foo_param = foo_param
        self.bar_param = bar_param

    def fit(self, X, Y):
        self.count += 1
        assert len(X) == len(Y)
        self.classes_ = np.unique(Y)
        return self

    def predict(self, T):
        return T.shape[0]

    predict_proba = predict
    predict_log_proba = predict
    decision_function = predict
    inverse_transform = predict

    def transform(self, X):
        return X

    def score(self, X=None, Y=None):
        if self.foo_param > 1:
            score = 1.0
        else:
            score = 0.0
        return score

    def get_params(self, deep=False):
        return {"foo_param": self.foo_param, "bar_param": self.bar_param}

    def set_params(self, **params):
        if "foo_param" in params:
            self.foo_param = params["foo_param"]
        if "bar_param" in params:
            self.bar_param = params["bar_param"]
        return self


class SleepClassifier(MockClassifier):
    def fit(self, X, Y):
        time.sleep(self.foo_param)
        return super().fit(X, Y)

    def partial_fit(self, X, Y):
        return self.fit(X, Y)

    def score(self, X=None, Y=None):
        return self.foo_param


class PlateauClassifier(MockClassifier):
    def __init__(self, foo_param=0, bar_param=0, converge_after=4):
        super(PlateauClassifier, self).__init__(foo_param, bar_param)
        self.converge_after = converge_after

    def fit(self, X, Y):
        return super().fit(X, Y)

    def partial_fit(self, X, Y):
        return self.fit(X, Y)

    def score(self, X=None, Y=None):
        if self.count >= self.converge_after:
            noise = 0.0
        else:
            noise = np.random.uniform(-0.5, 0.5)
        return self.foo_param + noise


class CheckingClassifier(BaseEstimator, ClassifierMixin):
    """Dummy classifier to test pipelining and meta-estimators.
    Checks some property of X and y in fit / predict.
    This allows testing whether pipelines / cross-validation or metaestimators
    changed the input.
    """

    def __init__(self,
                 check_y=None,
                 check_X=None,
                 foo_param=0,
                 expected_fit_params=None):
        self.check_y = check_y
        self.check_X = check_X
        self.foo_param = foo_param
        self.expected_fit_params = expected_fit_params

    def fit(self, X, y, **fit_params):
        assert len(X) == len(y)
        if self.check_X is not None:
            assert self.check_X(X)
        if self.check_y is not None:
            assert self.check_y(y)
        self.classes_ = np.unique(
            check_array(y, ensure_2d=False, allow_nd=True))
        if self.expected_fit_params:
            missing = set(self.expected_fit_params) - set(fit_params)
            assert (
                len(missing) == 0
            ), "Expected fit parameter(s) %s not " "seen." % list(missing)
            for key, value in fit_params.items():
                assert len(value) == len(X), (
                    "Fit parameter %s has length"
                    "%d; expected %d." % (key, len(value), len(X)))
        return self

    def predict(self, T):
        if self.check_X is not None:
            assert self.check_X(T)
        return self.classes_[np.zeros(_num_samples(T), dtype=np.int)]

    def score(self, X=None, Y=None):
        if self.foo_param > 1:
            score = 1.0
        else:
            score = 0.0
        return score


class BrokenClassifier(BaseEstimator):
    """Broken classifier that cannot be fit twice"""

    def __init__(self, parameter=None):
        self.parameter = parameter

    def fit(self, X, y):
        assert not hasattr(self, "has_been_fit_")
        self.has_been_fit_ = True

    def predict(self, X):
        return np.zeros(X.shape[0])


class ArraySlicingWrapper:
    def __init__(self, array):
        self.array = array

    def __getitem__(self, aslice):
        return MockDataFrame(self.array[aslice])


class MockDataFrame:
    # have shape and length but don't support indexing.
    def __init__(self, array):
        self.array = array
        self.values = array
        self.shape = array.shape
        self.ndim = array.ndim
        # ugly hack to make iloc work.
        self.iloc = ArraySlicingWrapper(array)

    def __len__(self):
        return len(self.array)

    def __array__(self, dtype=None):
        # Pandas data frames also are array-like: we want to make sure that
        # input validation in cross-validation does not try to call that
        # method.
        return self.array
