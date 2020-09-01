import unittest
import ray
from tune_sklearn._trainable import _Trainable
from tune_sklearn._detect_xgboost import has_xgboost

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv
from sklearn.svm import SVC


def create_xgboost():
    from xgboost.sklearn import XGBClassifier
    return XGBClassifier(
        learning_rate=0.02,
        n_estimators=5,
        objective="binary:logistic",
        nthread=4)


class TrainableTest(unittest.TestCase):
    X_id = None
    y_id = None
    X = None
    y = None

    @classmethod
    def setUpClass(cls):
        ray.init(local_mode=True)
        X, y = make_classification(
            n_samples=50, n_features=50, n_informative=3, random_state=0)
        cls.X_id, cls.y_id = ray.put(X), ray.put(y)
        cls.y = y
        cls.X = X

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def base_params(self, estimator_list):
        config = {"estimator_list": estimator_list}
        cv = check_cv(
            cv=len(estimator_list), y=self.y, classifier=estimator_list[0])
        config["X_id"] = self.X_id
        config["y_id"] = self.y_id
        config["early_stopping"] = False
        config["max_iters"] = 1
        config["groups"] = None
        config["cv"] = cv
        config["fit_params"] = None
        config["scoring"] = check_scoring(estimator_list[0], scoring=None)
        config["return_train_score"] = False
        config["n_jobs"] = 1
        return config

    def test_basic_train(self):
        config = self.base_params(estimator_list=[SVC(), SVC()])
        trainable = _Trainable(config)
        trainable.train()
        trainable.stop()

    @unittest.skipIf(not has_xgboost(), "xgboost not installed")
    def testXGBoostEarlyStop(self):

        config = self.base_params(
            estimator_list=[create_xgboost(),
                            create_xgboost()])
        config["early_stopping"] = True
        trainable = _Trainable(config)
        trainable.train()
        assert all(trainable.saved_models)
        trainable.train()
        assert all(trainable.saved_models)
        trainable.stop()

    @unittest.skipIf(not has_xgboost(), "xgboost not installed")
    def testXGBoostNoEarlyStop(self):
        config = self.base_params(
            estimator_list=[create_xgboost(),
                            create_xgboost()])
        config["early_stopping"] = False
        trainable = _Trainable(config)
        trainable.train()
        assert not any(trainable.saved_models)
        trainable.stop()

    def testPartialFit(self):
        config = self.base_params([SGDClassifier(), SGDClassifier()])
        config["early_stopping"] = True
        trainable = _Trainable(config)
        trainable.train()
        assert trainable.estimator_list[0].t_ > 0
        previous_t = trainable.estimator_list[0].t_
        trainable.train()
        assert trainable.estimator_list[0].t_ > previous_t
        trainable.stop()

    def testNoPartialFit(self):
        config = self.base_params([SGDClassifier(), SGDClassifier()])
        config["early_stopping"] = False
        trainable = _Trainable(config)
        trainable.train()
        assert not hasattr(trainable.estimator_list[0], "t_")
        trainable.train()
        assert not hasattr(trainable.estimator_list[0], "t_")
        trainable.stop()

    def testWarmStart(self):
        # Hard to get introspection so we just test that it runs.
        config = self.base_params([LogisticRegression(), LogisticRegression()])
        config["early_stopping"] = True
        trainable = _Trainable(config)
        trainable.train()
        trainable.train()
        trainable.stop()
