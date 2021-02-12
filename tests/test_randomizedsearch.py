import time

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.datasets import make_classification, make_regression
from sklearn.decomposition import PCA
from scipy.stats import expon
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn import datasets
from skopt.space.space import Real
from ray import tune

from ray.tune.schedulers import MedianStoppingRule
import unittest
from unittest.mock import patch
import os

from tune_sklearn import TuneSearchCV
from tune_sklearn._detect_booster import (has_xgboost, has_catboost,
                                          has_required_lightgbm_version)
from tune_sklearn.utils import EarlyStopping
from test_utils import SleepClassifier, PlateauClassifier


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
            "training_iteration",
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
                    if not key.startswith("rank")
                    and key != "training_iteration"))
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

    def test_multi_best_classification(self):
        digits = datasets.load_digits()
        x = digits.data
        y = digits.target
        model = SGDClassifier()

        parameter_grid = {"alpha": [1e-4, 1e-1, 1], "epsilon": [0.01, 0.1]}
        scoring = ("accuracy", "f1_micro")
        search_methods = ["random", "bayesian", "hyperopt", "bohb", "optuna"]
        for search_method in search_methods:
            if search_method == "bohb":
                print("bobh test currently failing")
                continue

            tune_search = TuneSearchCV(
                model,
                parameter_grid,
                scoring=scoring,
                search_optimization=search_method,
                cv=2,
                n_trials=3,
                n_jobs=1,
                refit="accuracy")
            tune_search.fit(x, y)
            self.assertAlmostEqual(
                tune_search.best_score_,
                max(tune_search.cv_results_["mean_test_accuracy"]),
                places=10)

            p = tune_search.cv_results_["params"]
            scores = tune_search.cv_results_["mean_test_accuracy"]
            cv_best_param = max(
                list(zip(scores, p)), key=lambda pair: pair[0])[1]
            self.assertEqual(tune_search.best_params_, cv_best_param)

    def test_multi_best_classification_scoring_dict(self):
        digits = datasets.load_digits()
        x = digits.data
        y = digits.target
        model = SGDClassifier()

        parameter_grid = {"alpha": [1e-4, 1e-1, 1], "epsilon": [0.01, 0.1]}
        scoring = {"acc": "accuracy", "f1": "f1_micro"}
        search_methods = ["random", "bayesian", "hyperopt", "bohb", "optuna"]
        for search_method in search_methods:
            if search_method == "bohb":
                print("bobh test currently failing")
                continue
            tune_search = TuneSearchCV(
                model,
                parameter_grid,
                scoring=scoring,
                search_optimization=search_method,
                cv=2,
                n_trials=3,
                n_jobs=1,
                refit="acc")
            tune_search.fit(x, y)
            self.assertAlmostEqual(
                tune_search.best_score_,
                max(tune_search.cv_results_["mean_test_acc"]),
                places=10)

            p = tune_search.cv_results_["params"]
            scores = tune_search.cv_results_["mean_test_acc"]
            cv_best_param = max(
                list(zip(scores, p)), key=lambda pair: pair[0])[1]
            self.assertEqual(tune_search.best_params_, cv_best_param)

    def test_multi_best_regression(self):
        x, y = make_regression(n_samples=100, n_features=10, n_informative=5)
        model = SGDRegressor()
        parameter_grid = {"alpha": [1e-4, 1e-1, 1], "epsilon": [0.01, 0.1]}

        scoring = ("neg_mean_absolute_error", "neg_mean_squared_error")

        search_methods = ["random", "bayesian", "hyperopt", "bohb", "optuna"]
        for search_method in search_methods:
            if search_method == "bohb":
                print("bobh test currently failing")
                continue
            tune_search = TuneSearchCV(
                model,
                parameter_grid,
                scoring=scoring,
                search_optimization=search_method,
                cv=2,
                n_trials=3,
                n_jobs=1,
                refit="neg_mean_absolute_error")
            tune_search.fit(x, y)
            self.assertAlmostEqual(
                tune_search.best_score_,
                max(tune_search.cv_results_[
                    "mean_test_neg_mean_absolute_error"]),
                places=10)

            p = tune_search.cv_results_["params"]
            scores = tune_search.cv_results_[
                "mean_test_neg_mean_absolute_error"]
            cv_best_param = max(
                list(zip(scores, p)), key=lambda pair: pair[0])[1]
            self.assertEqual(tune_search.best_params_, cv_best_param)

    def test_multi_refit_false(self):
        digits = datasets.load_digits()
        x = digits.data
        y = digits.target
        model = SGDClassifier()

        parameter_grid = {"alpha": [1e-4, 1e-1, 1], "epsilon": [0.01, 0.1]}
        scoring = ("accuracy", "f1_micro")

        tune_search = TuneSearchCV(
            model,
            parameter_grid,
            scoring=scoring,
            search_optimization="random",
            cv=2,
            n_trials=3,
            n_jobs=1,
            refit=False)
        tune_search.fit(x, y)
        with self.assertRaises(ValueError) as exc:
            tune_search.best_score_
        self.assertTrue(("instance was initialized with refit=False. "
                         "For multi-metric evaluation,") in str(exc.exception))
        with self.assertRaises(ValueError) as exc:
            tune_search.best_index_
        self.assertTrue(("instance was initialized with refit=False. "
                         "For multi-metric evaluation,") in str(exc.exception))
        with self.assertRaises(ValueError) as exc:
            tune_search.best_params_
        self.assertTrue(("instance was initialized with refit=False. "
                         "For multi-metric evaluation,") in str(exc.exception))

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

    @unittest.skipIf(not has_xgboost(), "xgboost not installed")
    def test_early_stop_xgboost_pipeline(self):
        from xgboost.sklearn import XGBClassifier
        from sklearn.pipeline import Pipeline
        TuneSearchCV(
            Pipeline([("model", XGBClassifier())]), {"model__C": [1, 2]},
            early_stopping=True,
            pipeline_auto_early_stop=True,
            cv=2,
            n_trials=2,
            max_iters=10)

    @unittest.skipIf(not has_required_lightgbm_version(),
                     "lightgbm not installed")
    def test_early_stop_lightgbm_pipeline(self):
        from lightgbm import LGBMClassifier
        from sklearn.pipeline import Pipeline
        TuneSearchCV(
            Pipeline([("model", LGBMClassifier())]),
            {"model__learning_rate": [0.1, 0.5]},
            early_stopping=True,
            pipeline_auto_early_stop=True,
            cv=2,
            n_trials=2,
            max_iters=10)

    @unittest.skipIf(not has_catboost(), "catboost not installed")
    def test_early_stop_catboost_pipeline(self):
        from catboost import CatBoostClassifier
        from sklearn.pipeline import Pipeline
        TuneSearchCV(
            Pipeline([("model", CatBoostClassifier())]),
            {"model__learning_rate": [0.1, 0.5]},
            early_stopping=True,
            pipeline_auto_early_stop=True,
            cv=2,
            n_trials=2,
            max_iters=10)

    def test_max_iters(self):
        X, y = make_classification(
            n_samples=50, n_features=50, n_informative=3, random_state=0)

        clf = PlateauClassifier(converge_after=20)

        search = TuneSearchCV(
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

        search = TuneSearchCV(
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
        X, y = make_classification(
            n_samples=50, n_features=50, n_informative=3, random_state=0)

        clf = SleepClassifier()
        # SleepClassifier sleeps for `foo_param` seconds, `cv` times.
        # Thus, the time budget is exhausted after testing the first two
        # `foo_param`s.
        search = TuneSearchCV(
            clf, {"foo_param": [1.1, 1.2, 2.5]},
            time_budget_s=5.0,
            cv=2,
            max_iters=5,
            early_stopping=True)

        start = time.time()
        search.fit(X, y)
        taken = time.time() - start

        print(search)
        # Without timeout we would need over 50 seconds for this to
        # finish. Allow for some initialization overhead
        self.assertLess(taken, 25.0)


class TestSearchSpace(unittest.TestCase):
    def setUp(self):
        self.clf = SGDClassifier(alpha=1, epsilon=0.1, penalty="l2")
        self.parameter_grid = {
            "alpha": tune.uniform(1e-4, 0.5),
            "epsilon": tune.uniform(0.01, 0.05),
            "penalty": tune.choice(["elasticnet", "l1"]),
        }

    def testRandom(self):
        self._test_method("random")

    def testBayesian(self):
        self._test_method("bayesian")

    def testHyperopt(self):
        self._test_method("hyperopt")

    @unittest.skip("bohb test currently failing")
    def testBohb(self):
        self._test_method("bohb")

    def testOptuna(self):
        self._test_method("optuna")

    def _test_method(self, search_method, **kwargs):
        digits = datasets.load_digits()
        x = digits.data
        y = digits.target

        tune_search = TuneSearchCV(
            self.clf,
            self.parameter_grid,
            search_optimization=search_method,
            cv=2,
            n_trials=3,
            n_jobs=1,
            refit=True,
            **kwargs)
        tune_search.fit(x, y)
        self.assertEquals(len(tune_search.cv_results_["params"]), 3)
        params = tune_search.best_estimator_.get_params()
        print({
            k: v
            for k, v in params.items() if k in ("alpha", "epsilon", "penalty")
        })
        self.assertTrue(1e-4 <= params["alpha"] <= 0.5)
        self.assertTrue(0.01 <= params["epsilon"] <= 0.05)
        self.assertTrue(params["penalty"] in ("elasticnet", "l1"))
        return tune_search

    def _test_points_to_evaluate(self, search_method):
        points = [{
            "alpha": 0.4,
            "epsilon": 0.01,
            "penalty": "elasticnet"
        }, {
            "alpha": 0.3,
            "epsilon": 0.02,
            "penalty": "l1"
        }]
        try:
            results = self._test_method(
                search_method, points_to_evaluate=points)
        except TypeError:
            self.skipTest(f"The current version of Ray does not support the "
                          f"`points_to_evaluate` argument for search method "
                          f"`{search_method}`. Skipping test.")
            return

        for i in range(len(points)):
            trial_config = results.cv_results_["params"][i]
            trial_config_dict = {
                k: trial_config[k]
                for k in self.parameter_grid
            }
            try:
                self.assertDictEqual(trial_config_dict, points[i])
            except AssertionError:
                # The latest master does not LIFO to FIFO conversion.
                # Todo(krfricke): Remove when merged:
                # https://github.com/ray-project/ray/pull/12790
                if search_method == "hyperopt":
                    self.assertDictEqual(
                        trial_config_dict,
                        points[0 if i == 1 else 1  # Reverse order
                               ])
                else:
                    raise

    def testBayesianPointsToEvaluate(self):
        self._test_points_to_evaluate("bayesian")

    def testHyperoptPointsToEvaluate(self):
        from ray.tune.suggest.hyperopt import HyperOptSearch
        # Skip test if category conversion is not available
        if not hasattr(HyperOptSearch, "_convert_categories_to_indices"):
            self.skipTest(f"The current version of Ray does not support the "
                          f"`points_to_evaluate` argument for search method "
                          f"`hyperopt`. Skipping test.")
            return
        self._test_points_to_evaluate("hyperopt")

    @unittest.skip("bohb currently failing not installed")
    def testBOHBPointsToEvaluate(self):
        self._test_points_to_evaluate("bohb")

    def testOptunaPointsToEvaluate(self):
        self._test_points_to_evaluate("optuna")

    def _test_seed_run(self, search_optimization, seed):
        digits = datasets.load_digits()

        x = digits.data
        y = digits.target

        parameters = {
            "classify__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1],
            "classify__epsilon": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        }

        pipe = Pipeline([("reduce_dim", PCA()), ("classify", SGDClassifier())])

        if isinstance(seed, str):
            _seed = np.random.RandomState(seed=int(seed))
        else:
            _seed = seed
        tune_search_1 = TuneSearchCV(
            pipe,
            parameters.copy(),
            early_stopping=True,
            max_iters=1,
            search_optimization=search_optimization,
            random_state=_seed)
        tune_search_1.fit(x, y)

        if isinstance(seed, str):
            _seed = np.random.RandomState(seed=int(seed))
        else:
            _seed = seed
        tune_search_2 = TuneSearchCV(
            pipe,
            parameters.copy(),
            early_stopping=True,
            max_iters=1,
            search_optimization=search_optimization,
            random_state=_seed)
        tune_search_2.fit(x, y)

        try:
            self.assertSequenceEqual(tune_search_1.cv_results_["params"],
                                     tune_search_2.cv_results_["params"])
        except AssertionError:
            print(f"Seeds: {tune_search_1.seed} == {tune_search_2.seed}?")
            raise

    def test_seed_random(self):
        self._test_seed_run("random", seed=1234)
        self._test_seed_run("random", seed="1234")

    def test_seed_bayesian(self):
        self._test_seed_run("bayesian", seed=1234)
        self._test_seed_run("bayesian", seed="1234")

    @unittest.skip("BOHB is currently failing")
    def test_seed_bohb(self):
        self._test_seed_run("bohb", seed=1234)
        self._test_seed_run("bohb", seed="1234")

    def test_seed_hyperopt(self):
        self._test_seed_run("hyperopt", seed=1234)
        self._test_seed_run("hyperopt", seed="1234")

    def test_seed_optuna(self):
        self._test_seed_run("optuna", seed=1234)
        self._test_seed_run("optuna", seed="1234")


if __name__ == "__main__":
    unittest.main()
