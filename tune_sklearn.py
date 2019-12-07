"""
    A GridSearchCV interface built with a Ray Tune back-end.
    Implementation derived from referencing the equivalent
    GridSearchCV interfaces from Dask and Optuna.
    https://ray.readthedocs.io/en/latest/tune.html
    https://dask.org
    https://optuna.org
    -- Anthony Yu and Michael Chau
"""

from scipy.stats import _distn_infrastructure, rankdata
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import cross_validate, check_cv, ParameterGrid, ParameterSampler
from sklearn.model_selection._search import _check_param_grid
from sklearn.metrics.scorer import check_scoring
from sklearn.base import is_classifier
from sklearn.utils.metaestimators import _safe_split
from sklearn.base import clone
from ray import tune
from ray.tune import Trainable
from ray.tune.schedulers import PopulationBasedTraining, MedianStoppingRule
import numpy as np
import os
import pickle

# Helper class to train models
class _Trainable(Trainable):

    def _setup(self, config):

        self.estimator = clone(config.pop('estimator'))
        self.scheduler = config.pop('scheduler')
        self.X = config.pop('X')
        self.y = config.pop('y')
        self.groups = config.pop('groups')
        self.fit_params = config.pop('fit_params')
        self.scoring = config.pop('scoring')
        self.early_stopping = config.pop('early_stopping')
        self.iters = config.pop('iters')
        self.cv = config.pop('cv')
        self.return_train_score = config.pop('return_train_score')

        self.estimator_config = config

        if self.early_stopping:
            n_splits = self.cv.get_n_splits(self.X, self.y)
            self.fold_scores = np.zeros(n_splits)
            self.fold_train_scores = np.zeros(n_splits)
            for i in range(n_splits):
                self.estimator[i].set_params(**self.estimator_config)
        else:
            self.estimator.set_params(**self.estimator_config)

    def _train(self):
        if self.early_stopping:
            for i, (train, test) in enumerate(self.cv.split(self.X, self.y)):
                X_train, y_train = _safe_split(self.estimator, self.X, self.y, train)
                X_test, y_test = _safe_split(
                    self.estimator,
                    self.X,
                    self.y,
                    test,
                    train_indices=train
                )
                self.estimator[i].partial_fit(X_train, y_train, np.unique(self.y))
                if self.return_train_score:
                    self.fold_train_scores[i] = self.scoring(self.estimator[i], X_train, y_train)
                self.fold_scores[i] = self.scoring(self.estimator[i], X_test, y_test)

            self.mean_scores = sum(self.fold_scores) / len(self.fold_scores)

            if self.return_train_score:
                self.mean_train_scores = sum(self.fold_train_scores) / len(self.fold_train_scores)
                return {"average_test_score": self.mean_scores, "average_train_score": self.mean_train_scores}

            return {"average_test_score": self.mean_scores}
        else:
            scores = cross_validate(
                self.estimator,
                self.X,
                self.y,
                cv=self.cv,
                fit_params=self.fit_params,
                groups=self.groups,
                scoring=self.scoring
            )
            self.test_accuracy = sum(scores["test_score"]) / len(scores["test_score"])
            if self.return_train_score:
                self.train_accuracy = sum(scores["train_score"]) / len(scores["train_score"])
                return {"average_test_score": self.test_accuracy, "average_train_score": self.train_accuracy}
            return {"average_test_score": self.test_accuracy}

    def _save(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        with open(path, 'wb') as f:
            pickle.dump(self.estimator, f)
        return path

    def _restore(self, checkpoint):
        with open(checkpoint, 'rb') as f:
            self.estimator = pickle.load(f)

    def reset_config(self, new_config):
        return True


# Base class for Randomized and Grid Search CV
class TuneBaseSearchCV(BaseEstimator):
    # TODO
    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    # TODO
    @property
    def best_params_(self):
        # only if refit true
        check_is_fitted(self, "cv_results_")
        self._check_if_refit("best_params_")
        return self.cv_results_["params"][self.best_index_]

    # TODO
    @property
    def best_score_(self):
        # only if refit true
        check_is_fitted(self, "cv_results_")
        self._check_if_refit("best_score_")
        return self.cv_results_["mean_test_score"][self.best_index_]

    # TODO
    @property
    def classes_(self):
        return self.best_estimator_.classes_

    # TODO
    @property
    def decision_function(self):
        check_is_fitted(self, "cv_results_")
        return self.best_estimator_.decision_function

    # TODO
    @property
    def inverse_transform(self):
        check_is_fitted(self, "cv_results_")
        return self.best_estimator_.inverse_transform

    # TODO
    @property
    def predict(self):
        check_is_fitted(self, "cv_results_")
        return self.best_estimator_.predict

    # TODO
    @property
    def predict_log_proba(self):
        check_is_fitted(self, "cv_results_")
        return self.best_estimator_.predict_log_proba

    # TODO
    @property
    def predict_proba(self):
        check_is_fitted(self, "cv_results_")
        return self.best_estimator_.predict_proba

    # TODO
    @property
    def transform(self):
        check_is_fitted(self, "cv_results_")
        return self.best_estimator_.transform

    def _check_params(self):
        if not hasattr(self.estimator, 'fit'):
            raise ValueError('estimator must be a scikit-learn estimator.')

        if self.early_stopping and not hasattr(self.estimator, 'partial_fit'):
            raise ValueError('estimator must support partial_fit.')

    def _check_if_refit(self, attr):
        if not self.refit:
            raise AttributeError(
                "'{}' is not a valid attribute with " "'refit=False'.".format(attr)
            )

    def __init__(self,
                 estimator,
                 scheduler=None,
                 scoring=None,
                 n_jobs=None,
                 cv=5,
                 refit=True,
                 verbose=0,
                 error_score='raise',
                 return_train_score=False,
                 early_stopping=False,
                 iters=1,
    ):
        self.estimator = estimator
        self.scheduler = scheduler
        self.cv = cv
        self.scoring = check_scoring(estimator, scoring)
        self.n_jobs = n_jobs
        self.refit = refit
        self.verbose = verbose
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.early_stopping = early_stopping
        self.iters = iters

    def _get_param_iterator(self):
        raise NotImplementedError("Implement in a child class.")

    def fit(self, X, y=None, groups=None, **fit_params):
        self._check_params()
        classifier = is_classifier(self.estimator)
        cv = check_cv(self.cv, y, classifier)
        n_splits = cv.get_n_splits(X, y, groups)
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)
        resources_per_trial = None
        if self.n_jobs:
            resources_per_trial = {'cpu': self.n_jobs, 'gpu': 0}

        config = {}
        config['scheduler'] = self.scheduler
        config['X'] = X
        config['y'] = y
        config['groups'] = groups
        config['cv'] = cv
        config['fit_params'] = fit_params
        config['scoring'] = self.scoring
        config['early_stopping'] = self.early_stopping
        config['iters'] = self.iters
        config['return_train_score'] = self.return_train_score

        candidate_params = list(self._get_param_iterator())
        n_samples = n_splits * len(candidate_params)

        if self.early_stopping:
            config['estimator'] = [clone(self.estimator) for _ in range(cv.get_n_splits(X, y))]
            analysis = tune.run(
                    _Trainable,
                    scheduler=self.scheduler,
                    reuse_actors=True,
                    verbose=self.verbose,
                    stop={"training_iteration":self.iters},
                    num_samples=n_samples,
                    config=config,
                    checkpoint_at_end=True,
                    resources_per_trial=resources_per_trial,
                    )
        else:
            config['estimator'] = self.estimator
            analysis = tune.run(
                    _Trainable,
                    scheduler=self.scheduler,
                    reuse_actors=True,
                    verbose=self.verbose,
                    stop={"training_iteration":self.iters},
                    num_samples=n_samples,
                    config=config,
                    checkpoint_at_end=True,
                    resources_per_trial=resources_per_trial,
                    )

        self.cv_results_ = self._format_results(candidate_params, n_splits, analysis)

        if self.refit:
            self.best_index_ = np.flatnonzero(
                self.cv_results_["rank_test_score"] == 1
            )[0]
            self.best_estimator_ = clone(self.estimator)
            self.best_estimator_.set_params(**self.best_params_)
            self.best_estimator_.fit(X, y, **fit_params)

        return self

    def score(self, X, y=None):
        # only if refit true
        return self.scorer_(self.best_estimator_, X, y)

    def _format_results(self, candidate_params, n_splits, out):
        # TODO: Extract relevant parts out of `analysis` object from Tune
        dfs = out.trial_dataframes
        if self.return_train_score:
            arrays = [zip(*(df[df["done"] == True][["average_test_score", "average_train_score", "time_total_s"]].to_numpy()))
                      for df in dfs.values()]
            test_scores, train_scores, fit_times = zip(*arrays)
        else:
            arrays = [zip(*(df[df["done"] == True][["average_test_score", "time_total_s"]].to_numpy())) for df in dfs.values()]
            test_scores, fit_times = zip(*arrays)

        results = {"params": candidate_params}
        n_candidates = len(candidate_params)

        def _store(
            results,
            key_name,
            array,
            n_splits,
            n_candidates,
            weights=None,
            splits=False,
            rank=False,
        ):
            """A small helper to store the scores/times to the cv_results_"""
            # When iterated first by n_splits and then by parameters
            array = np.array(array, dtype=np.float64).reshape(n_splits, n_candidates).T
            if splits:
                for split_i in range(n_splits):
                    results["split%d_%s" % (split_i, key_name)] = array[:, split_i]

            array_means = np.average(array, axis=1, weights=weights)
            results["mean_%s" % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(
                np.average((array - array_means[:, np.newaxis]) ** 2, axis=1, weights=weights)
            )
            results["std_%s" % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method="min"), dtype=np.int32
                )

        _store(results, 'fit_time', fit_times, n_splits, n_candidates)
        _store(results, "test_score", test_scores, n_splits, n_candidates, splits=True, rank=True)
        if self.return_train_score:
            _store(results, "train_score", train_scores, n_splits, n_candidates, splits=True, rank=True)
        return results


class TuneRandomizedSearchCV(TuneBaseSearchCV):
    def __init__(self,
                 estimator,
                 param_distributions,
                 num_samples=5,
                 random_state=None,
                 scheduler=None,
                 scoring=None,
                 n_jobs=None,
                 cv=5,
                 refit=True,
                 verbose=0,
                 error_score='raise',
                 return_train_score=False,
                 early_stopping=False,
                 iters=1,
    ):
        super(TuneRandomizedSearchCV, self).__init__(
            estimator=estimator,
            scheduler=scheduler,
            scoring=scoring,
            n_jobs=n_jobs,
            cv=cv,
            refit=refit,
            error_score=error_score,
            return_train_score=return_train_score,
            early_stopping=early_stopping,
            iters=iters,
        )

        self.param_distributions = param_distributions
        self.num_samples = num_samples
        self.random_state = random_state

    def _get_param_iterator(self):
        return ParameterSampler(self.param_distributions, self.num_samples, random_state=self.random_state)


class TuneGridSearchCV(TuneBaseSearchCV):
    def __init__(self,
                 estimator,
                 param_grid,
                 scheduler=None,
                 scoring=None,
                 n_jobs=None,
                 cv=5,
                 refit=True,
                 verbose=0,
                 error_score='raise',
                 return_train_score=False,
                 early_stopping=False,
                 iters=1,
    ):
        super(TuneGridSearchCV, self).__init__(
            estimator=estimator,
            scheduler=scheduler,
            scoring=scoring,
            n_jobs=n_jobs,
            cv=cv,
            refit=refit,
            error_score=error_score,
            return_train_score=return_train_score,
            early_stopping=early_stopping,
            iters=iters,
        )

        _check_param_grid(param_grid)
        self.param_grid = param_grid

    def _get_param_iterator(self):
        return ParameterGrid(self.param_grid)
