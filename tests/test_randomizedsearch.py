from tune_sklearn import TuneSearchCV
import numpy as np
from numpy.testing import assert_array_equal
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from scipy.stats import expon
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import datasets
from skopt.space.space import Real
from ray.tune.schedulers import MedianStoppingRule
import unittest
from unittest.mock import patch
import os
from tune_sklearn._detect_booster import (has_xgboost, has_catboost,
                                          has_required_lightgbm_version)
from tune_sklearn.utils import EarlyStopping


class RandomizedSearchTest(unittest.TestCase):
    def test_random_search_cv_results(self):
        # Make a dataset with a lot of noise to get various kind of prediction
        # errors across CV folds and parameter settings
        X, y = make_classification(
            n_samples=50, n_features=50, n_informative=3, random_state=0)

        # scipy.stats dists now supports `seed` but we still support scipy 0.12
        # which doesn't support the seed. Hence the assertions in the test for
        # random_search alone should not depend on randomization.
        n_splits = 3
        n_search_iter = 30
        params = dict(C=expon(scale=10), gamma=expon(scale=0.1))
        random_search = TuneSearchCV(
            SVC(),
            n_trials=n_search_iter,
            cv=n_splits,
            param_distributions=params,
            return_train_score=True,
            n_jobs=2)
        random_search.fit(X, y)

        param_keys = ("param_C", "param_gamma")
        score_keys = (
            "mean_test_score",
            "mean_train_score",
            "rank_test_score",
            "rank_train_score",
            "split0_test_score",
            "split1_test_score",
            "split2_test_score",
            "split0_train_score",
            "split1_train_score",
            "split2_train_score",
            "std_test_score",
            "std_train_score",
            "time_total_s",
        )
        n_cand = n_search_iter

        def test_check_cv_results_array_types(cv_results, param_keys,
                                              score_keys):
            # Check if the search `cv_results`'s array are of correct types
            self.assertTrue(
                all(
                    isinstance(cv_results[param], np.ma.MaskedArray)
                    for param in param_keys))
            self.assertTrue(
                all(cv_results[key].dtype == object for key in param_keys))
            self.assertFalse(
                any(
                    isinstance(cv_results[key], np.ma.MaskedArray)
                    for key in score_keys))
            self.assertTrue(
                all(cv_results[key].dtype == np.float64 for key in score_keys
                    if not key.startswith("rank")))
            self.assertEquals(cv_results["rank_test_score"].dtype, np.int32)

        def test_check_cv_results_keys(cv_results, param_keys, score_keys,
                                       n_cand):
            # Test the search.cv_results_ contains all the required results
            assert_array_equal(
                sorted(cv_results.keys()),
                sorted(param_keys + score_keys + ("params", )))
            self.assertTrue(
                all(cv_results[key].shape == (n_cand, )
                    for key in param_keys + score_keys))

        cv_results = random_search.cv_results_
        # Check results structure
        test_check_cv_results_array_types(cv_results, param_keys, score_keys)
        test_check_cv_results_keys(cv_results, param_keys, score_keys, n_cand)
        # For random_search, all the param array vals should be unmasked
        self.assertFalse(
            any(cv_results["param_C"].mask)
            or any(cv_results["param_gamma"].mask))

    def test_local_dir(self):
        digits = datasets.load_digits()
        x = digits.data
        y = digits.target

        clf = SGDClassifier()
        parameter_grid = {
            "alpha": Real(1e-4, 1e-1, 1),
            "epsilon": Real(0.01, 0.1)
        }

        scheduler = MedianStoppingRule(grace_period=10.0)

        tune_search = TuneSearchCV(
            clf,
            parameter_grid,
            early_stopping=scheduler,
            max_iters=10,
            local_dir="./test-result")
        tune_search.fit(x, y)

        self.assertTrue(len(os.listdir("./test-result")) != 0)

    def test_local_mode(self):
        digits = datasets.load_digits()
        x = digits.data
        y = digits.target

        clf = SGDClassifier()
        parameter_grid = {
            "alpha": Real(1e-4, 1e-1, 1),
            "epsilon": Real(0.01, 0.1)
        }
        tune_search = TuneSearchCV(
            clf,
            parameter_grid,
            n_jobs=1,
            max_iters=10,
            local_dir="./test-result")
        import ray
        with patch.object(ray, "init", wraps=ray.init) as wrapped_init:
            tune_search.fit(x, y)
        self.assertTrue(wrapped_init.call_args[1]["local_mode"])

    def test_multi_best(self):
        digits = datasets.load_digits()
        x = digits.data
        y = digits.target

        parameter_grid = {"alpha": [1e-4, 1e-1, 1], "epsilon": [0.01, 0.1]}

        scoring = ("accuracy", "f1_micro")

        tune_search = TuneSearchCV(
            SGDClassifier(),
            parameter_grid,
            scoring=scoring,
            max_iters=20,
            refit="accuracy")
        tune_search.fit(x, y)
        self.assertAlmostEqual(
            tune_search.best_score_,
            max(tune_search.cv_results_["mean_test_accuracy"]),
            places=10)

        p = tune_search.cv_results_["params"]
        scores = tune_search.cv_results_["mean_test_accuracy"]
        cv_best_param = max(list(zip(scores, p)), key=lambda pair: pair[0])[1]
        self.assertEqual(tune_search.best_params_, cv_best_param)

    def test_warm_start_detection(self):
        parameter_grid = {"alpha": Real(1e-4, 1e-1, 1)}
        from sklearn.ensemble import VotingClassifier, RandomForestClassifier
        clf = VotingClassifier(estimators=[(
            "rf", RandomForestClassifier(n_estimators=50, random_state=0))])
        tune_search = TuneSearchCV(
            clf,
            parameter_grid,
            n_jobs=1,
            max_iters=10,
            local_dir="./test-result")
        self.assertEqual(tune_search.early_stop_type,
                         EarlyStopping.NO_EARLY_STOP)

        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(random_state=0)
        tune_search2 = TuneSearchCV(
            clf,
            parameter_grid,
            n_jobs=1,
            max_iters=10,
            local_dir="./test-result")
        self.assertEqual(tune_search2.early_stop_type,
                         EarlyStopping.NO_EARLY_STOP)

        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
        tune_search3 = TuneSearchCV(
            clf,
            parameter_grid,
            n_jobs=1,
            max_iters=10,
            local_dir="./test-result")

        self.assertEqual(tune_search3.early_stop_type,
                         EarlyStopping.NO_EARLY_STOP)

        tune_search4 = TuneSearchCV(
            clf,
            parameter_grid,
            early_stopping=True,
            n_jobs=1,
            max_iters=10,
            local_dir="./test-result")
        self.assertEqual(tune_search4.early_stop_type,
                         EarlyStopping.WARM_START_ITER)

        clf = RandomForestClassifier()
        tune_search5 = TuneSearchCV(
            clf,
            parameter_grid,
            early_stopping=True,
            n_jobs=1,
            max_iters=10,
            local_dir="./test-result")
        self.assertEqual(tune_search5.early_stop_type,
                         EarlyStopping.WARM_START_ENSEMBLE)

    def test_warm_start_error(self):
        parameter_grid = {"alpha": Real(1e-4, 1e-1, 1)}
        from sklearn.ensemble import VotingClassifier, RandomForestClassifier
        clf = VotingClassifier(estimators=[(
            "rf", RandomForestClassifier(n_estimators=50, random_state=0))])
        tune_search = TuneSearchCV(
            clf,
            parameter_grid,
            n_jobs=1,
            early_stopping=False,
            max_iters=10,
            local_dir="./test-result")
        self.assertFalse(tune_search._can_early_stop())
        with self.assertRaises(ValueError):
            tune_search = TuneSearchCV(
                clf,
                parameter_grid,
                n_jobs=1,
                early_stopping=True,
                max_iters=10,
                local_dir="./test-result")

        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
        with self.assertRaises(ValueError):
            parameter_grid = {"max_iter": [1, 2]}
            TuneSearchCV(
                clf,
                parameter_grid,
                early_stopping=True,
                n_jobs=1,
                max_iters=10,
                local_dir="./test-result")

        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier()
        with self.assertRaises(ValueError):
            parameter_grid = {"n_estimators": [1, 2]}
            TuneSearchCV(
                clf,
                parameter_grid,
                early_stopping=True,
                n_jobs=1,
                max_iters=10,
                local_dir="./test-result")

    def test_warn_reduce_maxiters(self):
        parameter_grid = {"alpha": Real(1e-4, 1e-1, 1)}
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        with self.assertWarnsRegex(UserWarning, "max_iters is set"):
            TuneSearchCV(
                clf, parameter_grid, max_iters=10, local_dir="./test-result")
        with self.assertWarnsRegex(UserWarning, "max_iters is set"):
            TuneSearchCV(
                SGDClassifier(),
                parameter_grid,
                max_iters=10,
                local_dir="./test-result")

    def test_warn_early_stop(self):
        with self.assertWarnsRegex(UserWarning, "max_iters = 1"):
            TuneSearchCV(
                LogisticRegression(), {"C": [1, 2]}, early_stopping=True)
        with self.assertWarnsRegex(UserWarning, "max_iters = 1"):
            TuneSearchCV(
                SGDClassifier(), {"epsilon": [0.1, 0.2]}, early_stopping=True)

    @unittest.skipIf(not has_xgboost(), "xgboost not installed")
    def test_early_stop_xgboost_warn(self):
        from xgboost.sklearn import XGBClassifier
        with self.assertWarnsRegex(UserWarning, "github.com"):
            TuneSearchCV(
                XGBClassifier(), {"C": [1, 2]},
                early_stopping=True,
                max_iters=10)
        with self.assertWarnsRegex(UserWarning, "max_iters"):
            TuneSearchCV(
                XGBClassifier(), {"C": [1, 2]},
                early_stopping=True,
                max_iters=1)

    @unittest.skipIf(not has_required_lightgbm_version(),
                     "lightgbm not installed")
    def test_early_stop_lightgbm_warn(self):
        from lightgbm import LGBMClassifier
        with self.assertWarnsRegex(UserWarning, "lightgbm"):
            TuneSearchCV(
                LGBMClassifier(), {"learning_rate": [0.1, 0.5]},
                early_stopping=True,
                max_iters=10)
        with self.assertWarnsRegex(UserWarning, "max_iters"):
            TuneSearchCV(
                LGBMClassifier(), {"learning_rate": [0.1, 0.5]},
                early_stopping=True,
                max_iters=1)

    @unittest.skipIf(not has_catboost(), "catboost not installed")
    def test_early_stop_catboost_warn(self):
        from catboost import CatBoostClassifier
        with self.assertWarnsRegex(UserWarning, "Catboost"):
            TuneSearchCV(
                CatBoostClassifier(), {"learning_rate": [0.1, 0.5]},
                early_stopping=True,
                max_iters=10)
        with self.assertWarnsRegex(UserWarning, "max_iters"):
            TuneSearchCV(
                CatBoostClassifier(), {"learning_rate": [0.1, 0.5]},
                early_stopping=True,
                max_iters=1)

    def test_pipeline_early_stop(self):
        digits = datasets.load_digits()
        x = digits.data
        y = digits.target

        pipe = Pipeline([("reduce_dim", PCA()), ("classify", SGDClassifier())])
        parameter_grid = [
            {
                "classify__alpha": [1e-4, 1e-1, 1],
                "classify__epsilon": [0.01, 0.1]
            },
        ]

        with self.assertRaises(ValueError) as exc:
            TuneSearchCV(
                pipe,
                parameter_grid,
                early_stopping=True,
                pipeline_auto_early_stop=False,
                max_iters=10)
        self.assertTrue((
            "Early stopping is not supported because the estimator does "
            "not have `partial_fit`, does not support warm_start, or "
            "is a tree classifier. Set `early_stopping=False`."
        ) in str(exc.exception))

        tune_search = TuneSearchCV(
            pipe, parameter_grid, early_stopping=True, max_iters=10)
        tune_search.fit(x, y)


if __name__ == "__main__":
    unittest.main()
