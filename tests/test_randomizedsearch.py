from tune_sklearn.tune_search import TuneSearchCV
import numpy as np
from numpy.testing import assert_array_equal
from sklearn.datasets import make_classification
from scipy.stats import expon
from sklearn.svm import SVC
import unittest


class RandomizedSearchTest(unittest.TestCase):
    def test_random_search_cv_results(self):
        # Make a dataset with a lot of noise to get various kind of prediction
        # errors across CV folds and parameter settings
        X, y = make_classification(
            n_samples=200, n_features=100, n_informative=3, random_state=0)

        # scipy.stats dists now supports `seed` but we still support scipy 0.12
        # which doesn't support the seed. Hence the assertions in the test for
        # random_search alone should not depend on randomization.
        n_splits = 3
        n_search_iter = 30
        params = dict(C=expon(scale=10), gamma=expon(scale=0.1))
        random_search = TuneSearchCV(
            SVC(),
            n_iter=n_search_iter,
            cv=n_splits,
            param_distributions=params,
            return_train_score=True,
        )
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


if __name__ == "__main__":
    unittest.main()
