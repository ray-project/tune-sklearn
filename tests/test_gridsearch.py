import os
import time
import numpy as np
from numpy.testing import (
    assert_almost_equal,
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
)
from sklearn.metrics import f1_score, make_scorer
import pytest
import unittest
from unittest.mock import patch
from parameterized import parameterized
from sklearn.cluster import KMeans
from sklearn.svm import SVC, LinearSVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KernelDensity

from test_utils import (MockClassifier, CheckingClassifier, BrokenClassifier,
                        SleepClassifier, MockDataFrame, PlateauClassifier)
import ray
from ray import tune
from tune_sklearn import TuneGridSearchCV
from tune_sklearn import TuneSearchCV

# def test_check_cv_results_array_types(self, cv_results, param_keys,
#                                       score_keys):
#     # Check if the search `cv_results`'s array are of correct types
#     self.assertTrue(
#         all(
#             isinstance(cv_results[param], np.ma.MaskedArray)
#             for param in param_keys))
#     self.assertTrue(
#         all(cv_results[key].dtype == object for key in param_keys))
#     self.assertFalse(
#         any(
#             isinstance(cv_results[key], np.ma.MaskedArray)
#             for key in score_keys))
#     self.assertTrue(
#         all(cv_results[key].dtype == np.float64 for key in score_keys
#             if not key.startswith("rank")))
#     self.assertEquals(cv_results["rank_test_score"].dtype, np.int32)

# def test_check_cv_results_keys(self, cv_results, param_keys, score_keys,
#                                n_cand):
#     # Test the search.cv_results_ contains all the required results
#     assert_array_equal(
#         sorted(cv_results.keys()),
#         sorted(param_keys + score_keys + ("params", )))
#     self.assertTrue(
#         all(cv_results[key].shape == (n_cand, )
#             for key in param_keys + score_keys))


class LinearSVCNoScore(LinearSVC):
    """An LinearSVC classifier that has no score method."""

    @property
    def score(self):
        raise AttributeError


X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])


class GridSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    def tearDown(self):
        ray.shutdown()

    def test_grid_search(self):
        # Test that the best estimator contains the right value for foo_param
        clf = MockClassifier()
        grid_search = TuneGridSearchCV(clf, {"foo_param": [1, 2, 3]}, cv=3)
        # make sure it selects the smallest parameter in case of ties
        grid_search.fit(X, y)
        self.assertEqual(grid_search.best_estimator_.foo_param, 2)

        assert_array_equal(grid_search.cv_results_["param_foo_param"].data,
                           [1, 2, 3])

        # Smoke test the score etc:
        grid_search.score(X, y)
        grid_search.predict_proba(X)
        grid_search.decision_function(X)
        grid_search.transform(X)

        # Test exception handling on scoring
        grid_search.scoring = "sklearn"
        with self.assertRaises(ValueError):
            grid_search.fit(X, y)

    def test_grid_search_no_score(self):
        # Test grid-search on classifier that has no score function.
        clf = LinearSVC(random_state=0)
        X, y = make_blobs(random_state=0, centers=2)
        Cs = [0.1, 1, 10]
        clf_no_score = LinearSVCNoScore(random_state=0)

        # XXX: It seems there's some global shared state in LinearSVC - fitting
        # multiple `SVC` instances in parallel using threads sometimes results
        # in wrong results. This only happens with threads, not processes/sync.
        # For now, we'll fit using the sync scheduler.
        grid_search = TuneGridSearchCV(clf, {"C": Cs}, scoring="accuracy")
        grid_search.fit(X, y)

        grid_search_no_score = TuneGridSearchCV(
            clf_no_score, {"C": Cs}, scoring="accuracy")
        # smoketest grid search
        grid_search_no_score.fit(X, y)

        # check that best params are equal
        self.assertEqual(grid_search_no_score.best_params,
                         grid_search.best_params_)
        # check that we can call score and that it gives the correct result
        self.assertEqual(
            grid_search.score(X, y), grid_search_no_score.score(X, y))

        # giving no scoring function raises an error
        with self.assertRaises(TypeError) as exc:
            grid_search_no_score = TuneGridSearchCV(clf_no_score, {"C": Cs})

        self.assertTrue("no scoring" in str(exc.exception))

    @parameterized.expand([("grid", TuneGridSearchCV, {}), ("random",
                                                            TuneSearchCV, {
                                                                "n_trials": 1
                                                            })])
    def test_hyperparameter_searcher_with_fit_params(self, name, cls, kwargs):
        X = np.arange(100).reshape(10, 10)
        y = np.array([0] * 5 + [1] * 5)
        clf = CheckingClassifier(expected_fit_params=["spam", "eggs"])
        pipe = Pipeline([("clf", clf)])
        searcher = cls(pipe, {"clf__foo_param": [1, 2, 3]}, cv=2, **kwargs)

        # The CheckingClassifer generates an assertion error if
        # a parameter is missing or has length != len(X).
        with self.assertRaises(AssertionError) as exc:
            searcher.fit(X, y, clf__spam=np.ones(10))
        self.assertTrue("Expected fit parameter(s) ['eggs'] not seen." in str(
            exc.exception))

        searcher.fit(X, y, clf__spam=np.ones(10), clf__eggs=np.zeros(10))

    def test_grid_search_score_method(self):
        X, y = make_classification(
            n_samples=100, n_classes=2, flip_y=0.2, random_state=0)
        clf = LinearSVC(random_state=0)
        grid = {"C": [0.1]}

        search_no_scoring = TuneGridSearchCV(clf, grid, scoring=None).fit(X, y)
        search_accuracy = TuneGridSearchCV(
            clf, grid, scoring="accuracy").fit(X, y)
        search_no_score_method_auc = TuneGridSearchCV(
            LinearSVCNoScore(), grid, scoring="roc_auc").fit(X, y)
        search_auc = TuneGridSearchCV(clf, grid, scoring="roc_auc").fit(X, y)

        # Check warning only occurs in situation where behavior changed:
        # estimator requires score method to compete with scoring parameter
        score_no_scoring = search_no_scoring.score(X, y)
        score_accuracy = search_accuracy.score(X, y)
        score_no_score_auc = search_no_score_method_auc.score(X, y)
        score_auc = search_auc.score(X, y)

        # ensure the test is sane
        self.assertTrue(score_auc < 1.0)
        self.assertTrue(score_accuracy < 1.0)
        self.assertTrue(score_auc != score_accuracy)

        assert_almost_equal(score_accuracy, score_no_scoring)
        assert_almost_equal(score_auc, score_no_score_auc)

    def test_grid_search_groups(self):
        # Check if ValueError (when groups is None) propagates to
        # dcv.GridSearchCV
        # And also check if groups is correctly passed to the cv object
        rng = np.random.RandomState(0)

        X, y = make_classification(n_samples=15, n_classes=2, random_state=0)
        groups = rng.randint(0, 3, 15)

        clf = LinearSVC(random_state=0)
        grid = {"C": [1]}

        group_cvs = [
            LeaveOneGroupOut(),
            LeavePGroupsOut(2),
            GroupKFold(n_splits=3),
            GroupShuffleSplit(n_splits=3),
        ]
        for cv in group_cvs:
            gs = TuneGridSearchCV(clf, grid, cv=cv)
            with self.assertRaises(ValueError) as exc:
                gs.fit(X, y)
            self.assertTrue(
                "parameter should not be None" in str(exc.exception))

            gs.fit(X, y, groups=groups)

        non_group_cvs = [
            StratifiedKFold(n_splits=3),
            StratifiedShuffleSplit(n_splits=3)
        ]
        for cv in non_group_cvs:
            gs = TuneGridSearchCV(clf, grid, cv=cv)
            # Should not raise an error
            gs.fit(X, y)

    @pytest.mark.filterwarnings(
        "ignore::sklearn.exceptions.ConvergenceWarning")
    def test_classes__property(self):
        # Test that classes_ property matches best_estimator_.classes_
        X = np.arange(100).reshape(10, 10)
        y = np.array([0] * 5 + [1] * 5)
        Cs = [0.1, 1, 10]

        grid_search = TuneGridSearchCV(LinearSVC(random_state=0), {"C": Cs})
        grid_search.fit(X, y)
        assert_array_equal(grid_search.best_estimator_.classes_,
                           grid_search.classes_)

        # Test that regressors do not have a classes_ attribute
        grid_search = TuneGridSearchCV(Ridge(), {"alpha": [1.0, 2.0]})
        grid_search.fit(X, y)
        self.assertFalse(hasattr(grid_search, "classes_"))

        # Test that the grid searcher has no classes_ attribute before it's fit
        grid_search = TuneGridSearchCV(LinearSVC(random_state=0), {"C": Cs})
        self.assertFalse(hasattr(grid_search, "classes_"))

        # Test that the grid searcher has no classes_ attribute without a refit
        grid_search = TuneGridSearchCV(
            LinearSVC(random_state=0), {"C": Cs}, refit=False)
        grid_search.fit(X, y)
        self.assertFalse(hasattr(grid_search, "classes_"))

    def test_trivial_cv_results_attr(self):
        # Test search over a "grid" with only one point.
        # Non-regression test: grid_scores_ wouldn't be set by
        # dcv.GridSearchCV.
        clf = MockClassifier()
        grid_search = TuneGridSearchCV(clf, {"foo_param": [1]}, cv=3)
        grid_search.fit(X, y)
        self.assertTrue(hasattr(grid_search, "cv_results_"))

        random_search = TuneSearchCV(clf, {"foo_param": [0]}, n_trials=1, cv=3)
        random_search.fit(X, y)
        self.assertTrue(hasattr(random_search, "cv_results_"))

    def test_no_refit(self):
        # Test that GSCV can be used for model selection alone without
        # refitting
        clf = MockClassifier()
        grid_search = TuneGridSearchCV(
            clf, {"foo_param": [1, 2, 3]}, refit=False, cv=3)
        grid_search.fit(X, y)
        grid_search.best_index_
        grid_search.best_score_
        grid_search.best_params_

        # Make sure the predict/transform etc fns raise meaningful error msg
        for fn_name in (
                "best_estimator_",
                "predict",
                "predict_proba",
                "predict_log_proba",
                "transform",
                "inverse_transform",
        ):
            with self.assertRaises(NotFittedError) as exc:
                getattr(grid_search, fn_name)(X)
            self.assertTrue((
                "refit=False. %s is available only after refitting on the "
                "best parameters" % fn_name) in str(exc.exception))

    def test_grid_search_error(self):
        # Test that grid search will capture errors on data with different
        # length
        X_, y_ = make_classification(
            n_samples=200, n_features=100, random_state=0)

        clf = LinearSVC()
        cv = TuneGridSearchCV(clf, {"C": [0.1, 1.0]})
        with self.assertRaises(ValueError) as exc:
            cv.fit(X_[:180], y_)
        self.assertTrue(("Found input variables with inconsistent numbers of "
                         "samples: [180, 200]") in str(exc.exception))

    def test_grid_search_one_grid_point(self):
        X_, y_ = make_classification(
            n_samples=200, n_features=100, random_state=0)
        param_dict = {"C": [1.0], "kernel": ["rbf"], "gamma": [0.1]}

        clf = SVC()
        cv = TuneGridSearchCV(clf, param_dict)
        cv.fit(X_, y_)

        clf = SVC(C=1.0, kernel="rbf", gamma=0.1)
        clf.fit(X_, y_)

        assert_array_equal(clf.dual_coef_, cv.best_estimator_.dual_coef_)

    def test_grid_search_bad_param_grid(self):
        param_dict = {"C": 1.0}
        clf = SVC()

        with self.assertRaises(ValueError) as exc:
            TuneGridSearchCV(clf, param_dict)
        self.assertTrue(("Parameter grid for parameter (C) needs to"
                         " be a tune.grid_search, list or numpy array"
                         ) in str(exc.exception))

        param_dict = {"C": []}
        clf = SVC()

        with self.assertRaises(ValueError) as exc:
            TuneGridSearchCV(clf, param_dict)
        self.assertTrue((
            "Parameter values for parameter (C) need to be a non-empty "
            "sequence.") in str(exc.exception))

        param_dict = {"C": "1,2,3"}
        clf = SVC()

        with self.assertRaises(ValueError) as exc:
            TuneGridSearchCV(clf, param_dict)
        self.assertTrue(("Parameter grid for parameter (C) needs to"
                         " be a tune.grid_search, list or numpy array"
                         ) in str(exc.exception))

        param_dict = {"C": np.ones(6).reshape(3, 2)}
        clf = SVC()
        with self.assertRaises(ValueError):
            TuneGridSearchCV(clf, param_dict)

    def test_grid_search_sparse(self):
        # Test that grid search works with both dense and sparse matrices
        X_, y_ = make_classification(
            n_samples=200, n_features=100, random_state=0)

        clf = LinearSVC()
        cv = TuneGridSearchCV(clf, {"C": [0.1, 1.0]})
        cv.fit(X_[:180], y_[:180])
        y_pred = cv.predict(X_[180:])
        C = cv.best_estimator_.C

        X_ = sp.csr_matrix(X_)
        clf = LinearSVC()
        cv = TuneGridSearchCV(clf, {"C": [0.1, 1.0]})
        cv.fit(X_[:180].tocoo(), y_[:180])
        y_pred2 = cv.predict(X_[180:])
        C2 = cv.best_estimator_.C

        self.assertTrue(np.mean(y_pred == y_pred2) >= 0.9)
        self.assertEqual(C, C2)

    def test_grid_search_sparse_scoring(self):
        X_, y_ = make_classification(
            n_samples=200, n_features=100, random_state=0)

        clf = LinearSVC()
        cv = TuneGridSearchCV(clf, {"C": [0.1, 1.0]}, scoring="f1")
        cv.fit(X_[:180], y_[:180])
        y_pred = cv.predict(X_[180:])
        C = cv.best_estimator_.C

        X_ = sp.csr_matrix(X_)
        clf = LinearSVC()
        cv = TuneGridSearchCV(clf, {"C": [0.1, 1.0]}, scoring="f1")
        cv.fit(X_[:180], y_[:180])
        y_pred2 = cv.predict(X_[180:])
        C2 = cv.best_estimator_.C

        assert_array_equal(y_pred, y_pred2)
        self.assertEqual(C, C2)

        # test loss where greater is worse
        def f1_loss(y_true_, y_pred_):
            return -f1_score(y_true_, y_pred_)

        F1Loss = make_scorer(f1_loss, greater_is_better=False)
        cv = TuneGridSearchCV(clf, {"C": [0.1, 1.0]}, scoring=F1Loss)
        cv.fit(X_[:180], y_[:180])
        y_pred3 = cv.predict(X_[180:])
        C3 = cv.best_estimator_.C

        self.assertEqual(C, C3)
        assert_array_equal(y_pred, y_pred3)

    def test_grid_search_precomputed_kernel(self):
        # Test that grid search works when the input features are given in the
        # form of a precomputed kernel matrix
        X_, y_ = make_classification(
            n_samples=200, n_features=100, random_state=0)

        # compute the training kernel matrix corresponding to the linear kernel
        K_train = np.dot(X_[:180], X_[:180].T)
        y_train = y_[:180]

        clf = SVC(kernel="precomputed")
        cv = TuneGridSearchCV(clf, {"C": [0.1, 1.0]})
        cv.fit(K_train, y_train)

        self.assertTrue(cv.best_score_ >= 0)

        # compute the test kernel matrix
        K_test = np.dot(X_[180:], X_[:180].T)
        y_test = y_[180:]

        y_pred = cv.predict(K_test)

        self.assertTrue(np.mean(y_pred == y_test) >= 0)

        # test error is raised when the precomputed kernel is not array-like
        # or sparse
        # with self.assertRaises(TuneError):
        with self.assertRaises(ValueError) as exc:
            cv.fit(K_train.tolist(), y_train)
        self.assertTrue(("Precomputed kernels or affinity matrices have "
                         "to be passed as arrays or sparse matrices."
                         ) in str(exc.exception))

    def test_grid_search_precomputed_kernel_error_nonsquare(self):
        # Test that grid search returns an error with a non-square precomputed
        # training kernel matrix
        K_train = np.zeros((10, 20))
        y_train = np.ones((10, ))
        clf = SVC(kernel="precomputed")
        cv = TuneGridSearchCV(clf, {"C": [0.1, 1.0]})
        # with self.assertRaises(TuneError):
        with self.assertRaises(ValueError) as exc:
            cv.fit(K_train, y_train)
        self.assertTrue((
            "X should be a square kernel matrix") in str(exc.exception))

    def test_refit(self):
        # Regression test for bug in refitting
        # Simulates re-fitting a broken estimator; this used to break with
        # sparse SVMs.
        X = np.arange(100).reshape(10, 10)
        y = np.array([0] * 5 + [1] * 5)

        clf = TuneGridSearchCV(
            BrokenClassifier(), {"parameter": [0, 1]},
            scoring="accuracy",
            refit=True)
        clf.fit(X, y)

    def test_gridsearch_nd(self):
        # Pass X as list in dcv.GridSearchCV
        X_4d = np.arange(10 * 5 * 3 * 2).reshape(10, 5, 3, 2)
        y_3d = np.arange(10 * 7 * 11).reshape(10, 7, 11)
        clf = CheckingClassifier(
            check_X=lambda x: x.shape[1:] == (5, 3, 2),
            check_y=lambda x: x.shape[1:] == (7, 11),
        )
        grid_search = TuneGridSearchCV(clf, {"foo_param": [1, 2, 3]}, cv=3)
        grid_search.fit(X_4d, y_3d).score(X, y)
        self.assertTrue(hasattr(grid_search, "cv_results_"))

    def test_X_as_list(self):
        # Pass X as list in dcv.GridSearchCV
        X = np.arange(100).reshape(10, 10)
        y = np.array([0] * 5 + [1] * 5)

        clf = CheckingClassifier(check_X=lambda x: isinstance(x, list))
        cv = KFold(n_splits=3)
        grid_search = TuneGridSearchCV(clf, {"foo_param": [1, 2, 3]}, cv=cv)
        grid_search.fit(X.tolist(), y).score(X, y)
        self.assertTrue(hasattr(grid_search, "cv_results_"))

    def test_y_as_list(self):
        # Pass y as list in dcv.GridSearchCV
        X = np.arange(100).reshape(10, 10)
        y = np.array([0] * 5 + [1] * 5)

        clf = CheckingClassifier(check_y=lambda x: isinstance(x, list))
        cv = KFold(n_splits=3)
        grid_search = TuneGridSearchCV(clf, {"foo_param": [1, 2, 3]}, cv=cv)
        grid_search.fit(X, y.tolist()).score(X, y)
        self.assertTrue(hasattr(grid_search, "cv_results_"))

    @pytest.mark.filterwarnings("ignore")
    def test_pandas_input(self):
        # check cross_val_score doesn't destroy pandas dataframe
        types = [(MockDataFrame, MockDataFrame)]
        try:
            from pandas import Series, DataFrame

            types.append((DataFrame, Series))
        except ImportError:
            pass

        X = np.arange(100).reshape(10, 10)
        y = np.array([0] * 5 + [1] * 5)

        for InputFeatureType, TargetType in types:
            # X dataframe, y series
            X_df, y_ser = InputFeatureType(X), TargetType(y)
            clf = CheckingClassifier(
                check_X=lambda x: isinstance(x, InputFeatureType),
                check_y=lambda x: isinstance(x, TargetType),
            )

            grid_search = TuneGridSearchCV(clf, {"foo_param": [1, 2, 3]})
            grid_search.fit(X_df, y_ser).score(X_df, y_ser)
            grid_search.predict(X_df)
            self.assertTrue(hasattr(grid_search, "cv_results_"))

    def test_unsupervised_grid_search(self):
        # test grid-search with unsupervised estimator
        X, y = make_blobs(random_state=0)
        km = KMeans(random_state=0)
        grid_search = TuneGridSearchCV(
            km,
            param_grid=dict(n_clusters=[2, 3, 4]),
            scoring="adjusted_rand_score")
        grid_search.fit(X, y)
        # ARI can find the right number :)
        self.assertEqual(grid_search.best_params_["n_clusters"], 3)

        # Now without a score, and without y
        grid_search = TuneGridSearchCV(
            km, param_grid=dict(n_clusters=[2, 3, 4]))
        grid_search.fit(X)
        self.assertEqual(grid_search.best_params_["n_clusters"], 4)

    def test_gridsearch_no_predict(self):
        # test grid-search with an estimator without predict.
        # slight duplication of a test from KDE
        def custom_scoring(estimator, X):
            return 42 if estimator.bandwidth == 0.1 else 0

        X, _ = make_blobs(
            cluster_std=0.1, random_state=1, centers=[[0, 1], [1, 0], [0, 0]])
        search = TuneGridSearchCV(
            KernelDensity(),
            param_grid=dict(bandwidth=[0.01, 0.1, 1]),
            scoring=custom_scoring,
        )
        search.fit(X)
        self.assertEqual(search.best_params_["bandwidth"], 0.1)
        self.assertEqual(search.best_score_, 42)

    def test_gridsearch_multi_inputs(self):
        # Check that multimetric is detected
        parameter_grid = {"alpha": [1e-4, 1e-1, 1], "epsilon": [0.01, 0.1]}

        tune_search = TuneGridSearchCV(
            SGDClassifier(),
            parameter_grid,
            scoring=("accuracy", "f1_micro"),
            max_iters=20,
            cv=2,
            refit=False)
        tune_search.fit(X, y)
        self.assertTrue(tune_search.multimetric_)

        tune_search = TuneGridSearchCV(
            SGDClassifier(),
            parameter_grid,
            scoring="f1_micro",
            max_iters=20,
            cv=2)
        tune_search.fit(X, y)
        self.assertFalse(tune_search.multimetric_)

        # Make sure error is raised when refit isn't set properly
        tune_search = TuneGridSearchCV(
            SGDClassifier(),
            parameter_grid,
            scoring=("accuracy", "f1_micro"),
            cv=2,
            max_iters=20)
        with self.assertRaises(ValueError):
            tune_search.fit(X, y)

    def test_gridsearch_multi_cv_results(self):
        parameter_grid = {"alpha": [1e-4, 1e-1, 1], "epsilon": [0.01, 0.1]}

        scoring = ("accuracy", "f1_micro")
        cv = 2

        tune_search = TuneGridSearchCV(
            SGDClassifier(),
            parameter_grid,
            scoring=scoring,
            max_iters=20,
            refit=False,
            cv=cv)
        tune_search.fit(X, y)
        result = tune_search.cv_results_

        keys_to_check = []

        for s in scoring:
            keys_to_check.append("mean_test_%s" % s)
            for i in range(cv):
                keys_to_check.append("split%d_test_%s" % (i, s))

        for key in keys_to_check:
            self.assertIn(key, result)

    def test_gridsearch_no_multi_cv_results(self):
        parameter_grid = {"alpha": [1e-4, 1e-1, 1], "epsilon": [0.01, 0.1]}

        cv = 2

        tune_search = TuneGridSearchCV(
            SGDClassifier(), parameter_grid, max_iters=20, refit=False, cv=cv)
        tune_search.fit(X, y)
        result = tune_search.cv_results_

        keys_to_check = ["mean_test_score"]

        for i in range(cv):
            keys_to_check.append("split%d_test_score" % i)

        for key in keys_to_check:
            self.assertIn(key, result)

    def test_digits(self):
        # Loading the Digits dataset
        digits = datasets.load_digits()

        # To apply an classifier on this data, we need to flatten the image, to
        # turn the data in a (samples, feature) matrix:
        n_samples = len(digits.images)
        X = digits.images.reshape((n_samples, -1))
        y = digits.target

        # Split the dataset in two equal parts
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=0)

        # Set the parameters by cross-validation
        tuned_parameters = {
            "kernel": ["rbf"],
            "gamma": [1e-3, 1e-4],
            "C": [1, 10, 100, 1000]
        }

        tune_search = TuneGridSearchCV(SVC(), tuned_parameters)
        tune_search.fit(X_train, y_train)

        pred = tune_search.predict(X_test)
        print(pred)
        accuracy = np.count_nonzero(
            np.array(pred) == np.array(y_test)) / len(pred)
        print(accuracy)

    def test_diabetes(self):
        # load the diabetes datasets
        dataset = datasets.load_diabetes()
        X = dataset.data
        y = dataset.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=0)
        # prepare a range of alpha values to test
        alphas = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])
        param_grid = dict(alpha=alphas)
        # create and fit a ridge regression model, testing each alpha
        model = linear_model.Ridge()

        tune_search = TuneGridSearchCV(
            model,
            param_grid,
        )
        tune_search.fit(X_train, y_train)

        pred = tune_search.predict(X_test)
        print(pred)
        error = sum(np.array(pred) - np.array(y_test)) / len(pred)
        print(error)

    def test_local_mode(self):
        # Pass X as list in dcv.GridSearchCV
        X = np.arange(100).reshape(10, 10)
        y = np.array([0] * 5 + [1] * 5)

        clf = CheckingClassifier(check_X=lambda x: isinstance(x, list))
        cv = KFold(n_splits=3)
        with patch.object(ray, "init", wraps=ray.init) as wrapped_init:
            grid_search = TuneGridSearchCV(
                clf, {"foo_param": [1, 2, 3]}, n_jobs=1, cv=cv)
            grid_search.fit(X.tolist(), y).score(X, y)

        self.assertTrue(hasattr(grid_search, "cv_results_"))
        self.assertTrue(wrapped_init.call_args[1]["local_mode"])

    def test_tune_search_spaces(self):
        # Test mixed search spaces
        clf = MockClassifier()
        foo = [1, 2, 3]
        bar = [1, 2]
        grid_search = TuneGridSearchCV(
            clf, {
                "foo_param": tune.grid_search(foo),
                "bar_param": bar
            },
            refit=False,
            cv=3)
        grid_search.fit(X, y)
        params = grid_search.cv_results_["params"]
        results_grid = {k: {dic[k] for dic in params} for k in params[0]}
        self.assertTrue(len(results_grid["foo_param"]) == len(foo))
        self.assertTrue(len(results_grid["bar_param"]) == len(bar))

    def test_max_iters(self):
        X, y = make_classification(
            n_samples=50, n_features=50, n_informative=3, random_state=0)

        clf = PlateauClassifier(converge_after=20)

        search = TuneGridSearchCV(
            clf, {"foo_param": [2.0, 3.0, 4.0]},
            cv=2,
            max_iters=6,
            early_stopping=True)

        search.fit(X, y)

        print(search.cv_results_)

        for iters in search.cv_results_["training_iteration"]:
            # Stop after 6 iterations.
            self.assertLessEqual(iters, 6)

    def test_plateau(self):
        try:
            from ray.tune.stopper import TrialPlateauStopper
        except ImportError:
            self.skipTest("`TrialPlateauStopper` not available in "
                          "current Ray version.")
            return

        X, y = make_classification(
            n_samples=50, n_features=50, n_informative=3, random_state=0)

        clf = PlateauClassifier(converge_after=4)

        stopper = TrialPlateauStopper(metric="objective")

        search = TuneGridSearchCV(
            clf, {"foo_param": [2.0, 3.0, 4.0]},
            cv=2,
            max_iters=20,
            stopper=stopper,
            early_stopping=True)

        search.fit(X, y)

        print(search.cv_results_)

        for iters in search.cv_results_["training_iteration"]:
            # Converges after 4 iterations, but the stopper needs another
            # 4 to detect it converged.
            self.assertLessEqual(iters, 8)

    def test_timeout(self):
        clf = SleepClassifier()
        # SleepClassifier sleeps for `foo_param` seconds, `cv` times.
        # Thus, the time budget is exhausted after testing the first two
        # `foo_param`s.
        grid_search = TuneGridSearchCV(
            clf, {"foo_param": [1.1, 1.2, 2.5]},
            time_budget_s=5.0,
            cv=2,
            max_iters=5,
            early_stopping=True)

        start = time.time()
        grid_search.fit(X, y)
        taken = time.time() - start

        print(grid_search)
        # Without timeout we would need over 50 seconds for this to
        # finish. Allow for some initialization overhead
        self.assertLess(taken, 18.0)


if __name__ == "__main__":
    unittest.main()
