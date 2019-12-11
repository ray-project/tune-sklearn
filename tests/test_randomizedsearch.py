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
from sklearn.datasets import (
    make_blobs,
    make_classification,
    make_multilabel_classification,
)
from scipy.stats import expon
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from ray.tune.schedulers import PopulationBasedTraining, MedianStoppingRule
import random
import unittest

# TODO: Either convert to individual examples or to python unittests

class RandomizedSearchTest(unittest.TestCase):
    '''
    def test_random_forest(self):
        iris = datasets.load_iris()
        x = iris.data
        y = iris.target
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

        clf = RandomForestClassifier()
        param_grid = {
            'n_estimators': randint(20,80)
        }


        tune_search = TuneRandomizedSearchCV(clf, param_grid, scheduler=MedianStoppingRule(), iters=20)
        tune_search.fit(x_train, y_train)

        pred = tune_search.predict(x_test)
        print(pred)
        accuracy = np.count_nonzero(np.array(pred) == np.array(y_test))/len(pred)
        print(accuracy)

    def test_check_cv_results_array_types(self, cv_results, param_keys, score_keys):
        # Check if the search `cv_results`'s array are of correct types
        self.assertTrue(all(isinstance(cv_results[param], np.ma.MaskedArray) for param in param_keys))
        self.assertTrue(all(cv_results[key].dtype == object for key in param_keys))
        self.assertFalse(any(isinstance(cv_results[key], np.ma.MaskedArray) for key in score_keys))
        self.assertTrue(
        all(
            cv_results[key].dtype == np.float64
            for key in score_keys
            if not key.startswith("rank")
        )
        )
        self.assertEquals(cv_results["rank_test_score"].dtype, np.int32)

    def test_check_cv_results_keys(self, cv_results, param_keys, score_keys, n_cand):
        # Test the search.cv_results_ contains all the required results
        assert_array_equal(
            sorted(cv_results.keys()), sorted(param_keys + score_keys + ("params",))
        )
        self.assertTrue(all(cv_results[key].shape == (n_cand,) for key in param_keys + score_keys))

    def test_random_search_cv_results(self):
        # Make a dataset with a lot of noise to get various kind of prediction
        # errors across CV folds and parameter settings
        X, y = make_classification(
            n_samples=200, n_features=100, n_informative=3, random_state=0
        )

        # scipy.stats dists now supports `seed` but we still support scipy 0.12
        # which doesn't support the seed. Hence the assertions in the test for
        # random_search alone should not depend on randomization.
        n_splits = 3
        n_search_iter = 30
        params = dict(C=expon(scale=10), gamma=expon(scale=0.1))
        random_search = TuneRandomizedSearchCV(
            SVC(),
            iters=n_search_iter,
            cv=n_splits,
            #iid=False, #not yet supported
            param_distributions=params,
            return_train_score=True,
        )
        random_search.fit(X, y)
        random_search_iid = TuneRandomizedSearchCV(
            SVC(),
            iters=n_search_iter,
            cv=n_splits,
            #iid=True, #not yet supported
            param_distributions=params,
            return_train_score=True,
        )
        random_search_iid.fit(X, y)

        param_keys = ("param_C", "param_gamma")
        score_keys = (
            "mean_test_score",
            "mean_train_score",
            "rank_test_score",
            "split0_test_score",
            "split1_test_score",
            "split2_test_score",
            "split0_train_score",
            "split1_train_score",
            "split2_train_score",
            "std_test_score",
            "std_train_score",
            "mean_fit_time",
            "std_fit_time",
            "mean_score_time",
            "std_score_time",
        )
        n_cand = n_search_iter

        for search, iid in zip((random_search, random_search_iid), (False, True)):
            self.assertTrue(iid, search.iid)
            cv_results = search.cv_results_
            # Check results structure
            self.test_check_cv_results_array_types(cv_results, param_keys, score_keys)
            self.test_check_cv_results_keys(cv_results, param_keys, score_keys, n_cand)
            # For random_search, all the param array vals should be unmasked
            self.assertFalse(
            (
                any(cv_results["param_C"].mask) or any(cv_results["param_gamma"].mask)
            )
            )


            '''
    def test_pbt(self):
        iris = datasets.load_iris()
        x = iris.data
        y = iris.target
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

        clf = SGDClassifier()
        param_distributions = {
            'alpha': uniform(1e-4, 1e-1)
        }

        scheduler = PopulationBasedTraining(
                    time_attr="training_iteration",
                    metric="average_test_score",
                    mode="max",
                    perturbation_interval=5,
                    resample_probability=1.0,
                    hyperparam_mutations = {
                        "alpha" : lambda: np.random.choice([1e-4, 1e-3, 1e-2, 1e-1])
                    })

        tune_search = TuneRandomizedSearchCV(clf,
                    param_distributions,
                    scheduler=scheduler,
                    early_stopping=True,
                    iters=10,
                    verbose=1,
                    num_samples=3,
                    )
        tune_search.fit(x_train, y_train)

        pred = tune_search.predict(x_test)
        print(pred)
        accuracy = np.count_nonzero(np.array(pred) == np.array(y_test))/len(pred)
        print(accuracy)
        print(tune_search.best_params_)
'''
    def test_linear_iris(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        logistic = linear_model.LogisticRegression()

        # Create regularization penalty space
        penalty = ['l1', 'l2']

        # Create regularization hyperparameter space
        C = np.logspace(0, 4, 5)

        # Create hyperparameter options
        hyperparameters = dict(C=C, penalty=penalty)

        clf = TuneRandomizedSearchCV(logistic, hyperparameters, scheduler=MedianStoppingRule())
        clf.fit(X,y)

        pred = clf.predict(X)
        print(pred)
        accuracy = np.count_nonzero(np.array(pred) == np.array(y))/len(pred)
        print(accuracy)

'''

if __name__ == '__main__':
    unittest.main()

# TODO: Edit these tests from sklearn once we have properties/all signatures finished in tune_sklearn

'''
def test_grid_search():
    pass

def test_grid_search_with_fit_params():
    pass

def test_random_search_with_fit_params():
    pass

def test_grid_search_no_score():
    pass

def test_grid_search_score_method():
    pass

def test_grid_search_groups():
    pass

def test_no_refit():
    pass

def test_grid_search_error():
    pass

def test_grid_search_one_grid_point():
    pass
'''

