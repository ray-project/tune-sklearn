"""Parent class for a cross-validation interface
built with a Ray Tune back-end.

Implementation derived from referencing the equivalent
GridSearchCV interfaces from Dask and Optuna.

https://ray.readthedocs.io/en/latest/tune.html
https://dask.org
https://optuna.org
    -- Anthony Yu and Michael Chau
"""

from collections import defaultdict
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import check_cv
from sklearn.metrics import check_scoring
from sklearn.base import is_classifier
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
import ray
from ray.tune.schedulers import (
    PopulationBasedTraining, AsyncHyperBandScheduler, HyperBandScheduler,
    HyperBandForBOHB, MedianStoppingRule, TrialScheduler, ASHAScheduler)
import numpy as np
from numpy.ma import MaskedArray
import warnings


class TuneBaseSearchCV(BaseEstimator):
    """Abstract base class for TuneGridSearchCV and TuneSearchCV"""

    defined_schedulers = [
        "PopulationBasedTraining", "AsyncHyperBandScheduler",
        "HyperBandScheduler", "HyperBandForBOHB", "MedianStoppingRule",
        "ASHAScheduler"
    ]

    @property
    def _estimator_type(self):
        """str: Returns the estimator's estimator type, such as 'classifier'
        or 'regressor'.

        """
        return self.estimator._estimator_type

    @property
    def best_params_(self):
        """dict: Parameter setting that gave the best results on the hold
        out data.

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

        """
        self._check_if_refit("best_params_")
        return self.best_params

    @property
    def best_score_(self):
        """float: Mean cross-validated score of the best_estimator

        For multi-metric evaluation, this is present only if ``refit``
        is specified.

        """
        self._check_if_refit("best_score_")
        return self.best_score

    @property
    def classes_(self):
        """list: Get the list of unique classes found in the target `y`."""
        self._check_is_fitted("classes_")
        return self.best_estimator_.classes_

    @property
    def decision_function(self):
        """function: Get decision_function on the estimator with the best
        found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``decision_function``.

        """
        self._check_is_fitted("decision_function")
        return self.best_estimator_.decision_function

    @property
    def inverse_transform(self):
        """function: Get inverse_transform on the estimator with the best found
        parameters.

        Only available if the underlying estimator implements
        ``inverse_transform`` and ``refit=True``.

        """
        self._check_is_fitted("inverse_transform")
        return self.best_estimator_.inverse_transform

    @property
    def predict(self):
        """function: Get predict on the estimator with the best found
        parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.

        """
        self._check_is_fitted("predict")
        return self.best_estimator_.predict

    @property
    def predict_log_proba(self):
        """function: Get predict_log_proba on the estimator with the best found
        parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_log_proba``.

        """
        self._check_is_fitted("predict_log_proba")
        return self.best_estimator_.predict_log_proba

    @property
    def predict_proba(self):
        """function: Get predict_proba on the estimator with the best found
        parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_proba``.

        """
        self._check_is_fitted("predict_proba")
        return self.best_estimator_.predict_proba

    @property
    def transform(self):
        """function: Get transform on the estimator with the best found
        parameters.

        Only available if the underlying estimator supports ``transform`` and
        ``refit=True``.

        """
        self._check_is_fitted("transform")
        return self.best_estimator_.transform

    def _check_params(self):
        """Helper method to see if parameters passed in are valid.

        Raises:
            ValueError: if parameters are invalid.

        """
        if not hasattr(self.estimator, "fit"):
            raise ValueError("estimator must be a scikit-learn estimator.")

    def _check_if_refit(self, attr):
        """Helper method to see if the requested property is available based
        on the `refit` argument.

        Args:
            attr (str): Attribute requested by the user.

        Raises:
            AttributeError: If `self.refit` is False.

        """
        if not self.refit:
            raise AttributeError("'{}' is not a valid attribute with "
                                 "'refit=False'.".format(attr))

    def _check_is_fitted(self, method_name):
        """Helper method to see if the estimator has been fitted.

        Args:
            method_name (str): String of the method name called from the user.

        Raises:
            NotFittedError: If the estimator has not been fitted.
            TypeError: If the estimator is invalid (i.e. doesn't implement
                the sklearn estimator interface).

        """
        if not self.refit:
            msg = ("This {0} instance was initialized with refit=False. {1} "
                   "is available only after refitting on the best "
                   "parameters.").format(type(self).__name__, method_name)
            raise NotFittedError(msg)
        else:
            check_is_fitted(self)

    def __init__(
            self,
            estimator,
            scheduler=None,
            scoring=None,
            n_jobs=None,
            cv=5,
            refit=True,
            verbose=0,
            error_score="raise",
            return_train_score=False,
            early_stopping=False,
            max_iters=10,
    ):
        self.estimator = estimator
        self.early_stopping = early_stopping

        if self.early_stopping and self._can_early_stop():
            self.max_iters = max_iters
            if isinstance(scheduler, str):
                if scheduler in TuneBaseSearchCV.defined_schedulers:
                    if scheduler == "PopulationBasedTraining":
                        self.scheduler = PopulationBasedTraining(
                            metric="average_test_score")
                    elif scheduler == "AsyncHyperBandScheduler":
                        self.scheduler = AsyncHyperBandScheduler(
                            metric="average_test_score")
                    elif scheduler == "HyperBandScheduler":
                        self.scheduler = HyperBandScheduler(
                            metric="average_test_score")
                    elif scheduler == "HyperBandForBOHB":
                        self.scheduler = HyperBandForBOHB(
                            metric="average_test_score")
                    elif scheduler == "MedianStoppingRule":
                        self.scheduler = MedianStoppingRule(
                            metric="average_test_score")
                    elif scheduler == "ASHAScheduler":
                        self.scheduler = ASHAScheduler(
                            metric="average_test_score")
                else:
                    raise ValueError("{} is not a defined scheduler. "
                                     "Check the list of available schedulers."
                                     .format(scheduler))
            elif isinstance(scheduler, TrialScheduler):
                self.scheduler = scheduler
                self.scheduler.metric = "average_test_score"
            elif scheduler is None:
                self.scheduler = ASHAScheduler(metric="average_test_score")
            else:
                raise TypeError("Scheduler must be a str or tune scheduler")
        else:
            warnings.warn("Early stopping is not enabled. "
                          "This may be because the estimator "
                          "does not have `partial_fit`")

            self.max_iters = 1
            self.scheduler = None

        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.verbose = verbose
        self.error_score = error_score
        self.return_train_score = return_train_score

    def fit(self, X, y=None, groups=None, **fit_params):
        """Run fit with all sets of parameters. ``tune.run`` is used to perform
        the fit procedure, which is put in a helper function ``_tune_run``.

        Args:
            X (:obj:`array-like` (shape = [n_samples, n_features])):
                Training vector, where n_samples is the number of samples and
                n_features is the number of features.
            y (:obj:`array-like` (shape = [n_samples] or
                [n_samples, n_output]), optional):
                Target relative to X for classification or regression;
                None for unsupervised learning.
            groups (:obj:`array-like` (shape (n_samples,)), optional):
                Group labels for the samples used while splitting the dataset
                into train/test set. Only used in conjunction with a "Group"
                `cv` instance (e.g., `GroupKFold`).
            **fit_params (:obj:`dict` of str):
                Parameters passed to the ``fit`` method of the estimator.

        Returns:
            :obj:`TuneBaseSearchCV` child instance, after fitting.

        """
        ray.init(ignore_reinit_error=True, configure_logging=False)

        self._check_params()
        classifier = is_classifier(self.estimator)
        cv = check_cv(cv=self.cv, y=y, classifier=classifier)
        self.n_splits = cv.get_n_splits(X, y, groups)
        self.scoring = check_scoring(self.estimator, scoring=self.scoring)
        resources_per_trial = None
        if self.n_jobs and self.n_jobs != -1:
            resources_per_trial = {"cpu": self.n_jobs, "gpu": 0}

        X_id = ray.put(X)
        y_id = ray.put(y)

        config = {}
        config["scheduler"] = self.scheduler
        config["X_id"] = X_id
        config["y_id"] = y_id
        config["groups"] = groups
        config["cv"] = cv
        config["fit_params"] = fit_params
        config["scoring"] = self.scoring
        config["early_stopping"] = self.early_stopping
        config["max_iters"] = self.max_iters
        config["return_train_score"] = self.return_train_score

        self._fill_config_hyperparam(config)
        analysis = self._tune_run(config, resources_per_trial)

        self.cv_results_ = self._format_results(self.n_splits, analysis)

        if self.refit:
            best_config = analysis.get_best_config(
                metric="average_test_score", mode="max")
            self.best_params = self._clean_config_dict(best_config)
            self.best_estimator_ = clone(self.estimator)
            self.best_estimator_.set_params(**self.best_params)
            self.best_estimator_.fit(X, y, **fit_params)

            df = analysis.dataframe(metric="average_test_score", mode="max")
            self.best_score = df["average_test_score"].iloc[df[
                "average_test_score"].idxmax()]

        ray.shutdown()

        return self

    def score(self, X, y=None):
        """Compute the score(s) of an estimator on a given test set.

        Args:
            X (:obj:`array-like` (shape = [n_samples, n_features])):
                Input data, where n_samples is the number of samples and
                n_features is the number of features.
            y (:obj:`array-like` (shape = [n_samples] or
                [n_samples, n_output]), optional):
                Target relative to X for classification or regression;
                None for unsupervised learning.

        Returns:
            float: computed score

        """
        return self.scoring(self.best_estimator_, X, y)

    def _can_early_stop(self):
        """Helper method to determine if it is possible to do early stopping.

        Only sklearn estimators with partial_fit can be early stopped.

        Returns:
            bool: if the estimator can early stop

        """
        return (hasattr(self.estimator, "partial_fit")
                and callable(getattr(self.estimator, "partial_fit", None)))

    def _fill_config_hyperparam(self, config):
        """Fill in the ``config`` dictionary with the hyperparameters.

        For RandomizedSearchCV, samples are pulled from the distribution
        to be saved in the ``config`` dictionary.
        For GridSearchCV, the list is directly saved in the ``config``
        dictionary.

        Implement this functionality in a child class.

        Args:
            config (:obj:`dict`): dictionary to be filled in as the
                configuration for `tune.run`.

        """
        raise NotImplementedError("Define in child class")

    def _tune_run(self, config, resources_per_trial):
        """Wrapper to call ``tune.run``. Implement this in a child class.

        Args:
            config (:obj:`dict`): dictionary to be passed in as the
                configuration for `tune.run`.
            resources_per_trial (:obj:`dict` of int): dictionary specifying the
                number of cpu's and gpu's to use to train the model.

        """
        raise NotImplementedError("Define in child class")

    def _clean_config_dict(self, config):
        """Helper to remove keys from the ``config`` dictionary returned from
        ``tune.run``.

        Args:
            config (:obj:`dict`): Dictionary of all hyperparameter
                configurations and extra output from ``tune.run``., Keys for
                hyperparameters are the hyperparameter variable names
                and the values are the numeric values set to those variables.

        Returns:
            config (:obj:`dict`): Dictionary of all hyperparameter
                configurations without the output from ``tune.run``., Keys for
                hyperparameters are the hyperparameter variable names
                and the values are the numeric values set to those variables.
        """
        for key in [
                "estimator",
                "scheduler",
                "X_id",
                "y_id",
                "groups",
                "cv",
                "fit_params",
                "scoring",
                "early_stopping",
                "max_iters",
                "return_train_score",
        ]:
            config.pop(key, None)
        return config

    def _format_results(self, n_splits, out):
        """Helper to generate the ``cv_results_`` dictionary.

        Args:
            n_splits (int): integer specifying the number of folds when doing
                cross-validation.
            out (:obj:`ExperimentAnalysis`): Object returned by `tune.run`.

        Returns:
            results (:obj:`dict`): Dictionary of results to use for the
                interface's ``cv_results_``.

        """
        dfs = list(out.fetch_trial_dataframes().values())
        finished = [df[df["done"]] for df in dfs]
        test_scores = [
            df[[
                col for col in dfs[0].columns
                if "split" in col and "test_score" in col
            ]].to_numpy() for df in finished
        ]
        if self.return_train_score:
            train_scores = [
                df[[
                    col for col in dfs[0].columns
                    if "split" in col and "train_score" in col
                ]].to_numpy() for df in finished
            ]
        else:
            train_scores = None

        configs = out.get_all_configs()
        candidate_params = [
            self._clean_config_dict(configs[config_key])
            for config_key in configs
        ]

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
            array = np.array(
                array, dtype=np.float64).reshape((n_candidates, n_splits))
            if splits:
                for split_i in range(n_splits):
                    results["split%d_%s" % (split_i,
                                            key_name)] = array[:, split_i]

            array_means = np.average(array, axis=1, weights=weights)
            results["mean_%s" % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(
                np.average(
                    (array - array_means[:, np.newaxis])**2,
                    axis=1,
                    weights=weights))
            results["std_%s" % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method="min"), dtype=np.int32)

        _store(
            results,
            "test_score",
            test_scores,
            n_splits,
            n_candidates,
            splits=True,
            rank=True,
        )
        if self.return_train_score:
            _store(
                results,
                "train_score",
                train_scores,
                n_splits,
                n_candidates,
                splits=True,
                rank=True,
            )

        results["time_total_s"] = np.array(
            [df["time_total_s"].to_numpy() for df in finished]).flatten()

        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(
            lambda: MaskedArray(
                np.empty(n_candidates),
                mask=True,
                dtype=object,
            )
        )
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)

        return results
