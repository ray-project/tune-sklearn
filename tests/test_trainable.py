import unittest
import ray
from ray import tune
from tune_sklearn._trainable import _Trainable
from tune_sklearn._detect_booster import (
    has_xgboost, has_required_lightgbm_version, has_catboost)

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import check_cv
from sklearn.svm import SVC

from tune_sklearn.utils import _check_multimetric_scoring, get_early_stop_type


def create_xgboost():
    from xgboost.sklearn import XGBClassifier
    return XGBClassifier(
        learning_rate=0.02,
        n_estimators=5,
        objective="binary:logistic",
        nthread=4)


def create_lightgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(learning_rate=0.02, n_estimators=5, n_jobs=4)


def create_catboost():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        learning_rate=0.02, n_estimators=5, thread_count=4)


class TrainableTest(unittest.TestCase):
    X = None
    y = None

    @classmethod
    def setUpClass(cls):
        ray.init(local_mode=True)
        X, y = make_classification(
            n_samples=50, n_features=50, n_informative=3, random_state=0)
        cls.y = y
        cls.X = X

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def base_params(self, estimator_list):
        config = {}
        cv = check_cv(
            cv=len(estimator_list), y=self.y, classifier=estimator_list[0])
        config["early_stopping"] = False
        config["early_stop_type"] = get_early_stop_type(
            estimator_list[0], False)
        config["max_iters"] = 1
        config["groups"] = None
        config["cv"] = cv
        config["fit_params"] = None
        config["scoring"], _ = _check_multimetric_scoring(
            estimator_list[0], scoring=None)
        config["return_train_score"] = False
        config["n_jobs"] = 1
        return config

    def test_basic_train(self):
        estimator_list = [SVC(), SVC()]
        config = self.base_params(estimator_list)
        trainable = tune.with_parameters(
            _Trainable, X=self.X, y=self.y,
            estimator_list=estimator_list)(config)
        trainable.train()
        trainable.stop()

    @unittest.skipIf(not has_xgboost(), "xgboost not installed")
    def testXGBoostEarlyStop(self):
        estimator_list = [create_xgboost(), create_xgboost()]
        config = self.base_params(estimator_list=estimator_list)
        config["early_stopping"] = True
        config["early_stop_type"] = get_early_stop_type(
            estimator_list[0], True)
        trainable = tune.with_parameters(
            _Trainable, X=self.X, y=self.y,
            estimator_list=estimator_list)(config)
        trainable.train()
        assert all(trainable.saved_models)
        trainable.train()
        assert all(trainable.saved_models)
        trainable.stop()

    @unittest.skipIf(not has_xgboost(), "xgboost not installed")
    def testXGBoostNoEarlyStop(self):
        estimator_list = [create_xgboost(), create_xgboost()]
        config = self.base_params(estimator_list=estimator_list)
        config["early_stopping"] = False
        trainable = tune.with_parameters(
            _Trainable, X=self.X, y=self.y,
            estimator_list=estimator_list)(config)
        trainable.train()
        assert not any(trainable.saved_models)
        trainable.stop()

    @unittest.skipIf(not has_required_lightgbm_version(),
                     "lightgbm>=3.0.0 not installed")
    def testLGBMEarlyStop(self):
        estimator_list = [create_lightgbm(), create_lightgbm()]
        config = self.base_params(estimator_list=estimator_list)
        config["early_stopping"] = True
        config["early_stop_type"] = get_early_stop_type(
            estimator_list[0], True)
        trainable = tune.with_parameters(
            _Trainable, X=self.X, y=self.y,
            estimator_list=estimator_list)(config)
        trainable.train()
        assert all(trainable.saved_models)
        trainable.train()
        assert all(trainable.saved_models)
        trainable.stop()

    @unittest.skipIf(not has_required_lightgbm_version(),
                     "lightgbm>=3.0.0 not installed")
    def testLGBMNoEarlyStop(self):
        estimator_list = [create_lightgbm(), create_lightgbm()]
        config = self.base_params(estimator_list=estimator_list)
        config["early_stopping"] = False
        trainable = tune.with_parameters(
            _Trainable, X=self.X, y=self.y,
            estimator_list=estimator_list)(config)
        trainable.train()
        assert not any(trainable.saved_models)
        trainable.stop()

    @unittest.skipIf(not has_catboost(), "catboost not installed")
    def testCatboostEarlyStop(self):
        estimator_list = [create_catboost(), create_catboost()]
        config = self.base_params(estimator_list=estimator_list)
        config["early_stopping"] = True
        config["early_stop_type"] = get_early_stop_type(
            estimator_list[0], True)
        trainable = tune.with_parameters(
            _Trainable, X=self.X, y=self.y,
            estimator_list=estimator_list)(config)
        trainable.train()
        assert all(trainable.saved_models)
        trainable.train()
        assert all(trainable.saved_models)
        trainable.stop()

    @unittest.skipIf(not has_catboost(), "catboost not installed")
    def testCatboostNoEarlyStop(self):
        estimator_list = [create_catboost(), create_catboost()]
        config = self.base_params(estimator_list=estimator_list)
        config["early_stopping"] = False
        trainable = tune.with_parameters(
            _Trainable, X=self.X, y=self.y,
            estimator_list=estimator_list)(config)
        trainable.train()
        assert not any(trainable.saved_models)
        trainable.stop()

    def testPartialFit(self):
        estimator_list = [SGDClassifier(), SGDClassifier()]
        config = self.base_params(estimator_list)
        config["early_stopping"] = True
        config["early_stop_type"] = get_early_stop_type(
            estimator_list[0], True)
        trainable = tune.with_parameters(
            _Trainable, X=self.X, y=self.y,
            estimator_list=estimator_list)(config)
        trainable.train()
        assert trainable.estimator_list[0].t_ > 0
        previous_t = trainable.estimator_list[0].t_
        trainable.train()
        assert trainable.estimator_list[0].t_ > previous_t
        trainable.stop()

    def testNoPartialFit(self):
        estimator_list = [SGDClassifier(), SGDClassifier()]
        config = self.base_params(estimator_list)
        config["early_stopping"] = False
        trainable = tune.with_parameters(
            _Trainable, X=self.X, y=self.y,
            estimator_list=estimator_list)(config)
        trainable.train()
        assert not hasattr(trainable.estimator_list[0], "t_")
        trainable.train()
        assert not hasattr(trainable.estimator_list[0], "t_")
        trainable.stop()

    def testWarmStart(self):
        # Hard to get introspection so we just test that it runs.
        estimator_list = [LogisticRegression(), LogisticRegression()]
        config = self.base_params(estimator_list)
        config["early_stopping"] = True
        config["early_stop_type"] = get_early_stop_type(
            estimator_list[0], True)
        trainable = tune.with_parameters(
            _Trainable, X=self.X, y=self.y,
            estimator_list=estimator_list)(config)
        trainable.train()
        trainable.train()
        trainable.stop()
