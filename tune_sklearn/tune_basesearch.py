"""Parent class for a cross-validation interface
built with a Ray Tune back-end.

Implementation derived from referencing the equivalent
GridSearchCV interfaces from Dask and Optuna.

https://ray.readthedocs.io/en/latest/tune.html
https://dask.org
https://optuna.org
    -- Anthony Yu and Michael Chau
"""
import logging
from collections import defaultdict

from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import check_cv
from sklearn.base import is_classifier
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

import numpy as np
from numpy.ma import MaskedArray
import pandas as pd
import warnings
import multiprocessing
import os
import inspect
import time
import numbers

import ray
from ray.tune.trial import Trial
from ray.tune.schedulers import (
    PopulationBasedTraining, AsyncHyperBandScheduler, HyperBandScheduler,
    MedianStoppingRule, TrialScheduler, ASHAScheduler, HyperBandForBOHB)
from ray.tune.logger import (TBXLogger, JsonLogger, CSVLogger, MLFLowLogger,
                             Logger)

from tune_sklearn.utils import (EarlyStopping, get_early_stop_type,
                                check_is_pipeline, _check_multimetric_scoring)
from tune_sklearn._detect_booster import is_lightgbm_model

logger = logging.getLogger(__name__)


def resolve_early_stopping(early_stopping, max_iters, metric_name):
    if isinstance(early_stopping, str):
        if early_stopping in TuneBaseSearchCV.defined_schedulers:
            if early_stopping == "PopulationBasedTraining":
                return PopulationBasedTraining(metric=metric_name, mode="max")
            elif early_stopping == "AsyncHyperBandScheduler":
                return AsyncHyperBandScheduler(
                    metric=metric_name, mode="max", max_t=max_iters)
            elif early_stopping == "HyperBandScheduler":
                return HyperBandScheduler(
                    metric=metric_name, mode="max", max_t=max_iters)
            elif early_stopping == "MedianStoppingRule":
                return MedianStoppingRule(metric=metric_name, mode="max")
            elif early_stopping == "ASHAScheduler":
                return ASHAScheduler(
                    metric=metric_name, mode="max", max_t=max_iters)
            elif early_stopping == "HyperBandForBOHB":
                return HyperBandForBOHB(
                    metric=metric_name, mode="max", max_t=max_iters)
        raise ValueError("{} is not a defined scheduler. "
                         "Check the list of available schedulers."
                         .format(early_stopping))
    elif isinstance(early_stopping, TrialScheduler):
        early_stopping._metric = metric_name
        early_stopping._mode = "max"
        return early_stopping
    else:
        raise TypeError("`early_stopping` must be a str, boolean, "
                        f"or tune scheduler. Got {type(early_stopping)}.")


def resolve_loggers(loggers):
    init_loggers = {JsonLogger, CSVLogger}
    if loggers is None:
        return list(init_loggers)

    if not isinstance(loggers, list):
        raise TypeError("`loggers` must be a list of str or tune loggers.")

    for log in loggers:
        if isinstance(log, str):
            if log == "tensorboard":
                init_loggers.add(TBXLogger)
            elif log == "csv":
                init_loggers.add(CSVLogger)
            elif log == "mlflow":
                init_loggers.add(MLFLowLogger)
            elif log == "json":
                init_loggers.add(JsonLogger)
            else:
                raise ValueError(("{} is not one of the defined loggers. " +
                                  str(TuneBaseSearchCV.defined_schedulers))
                                 .format(log))
        elif inspect.isclass(log) and issubclass(log, Logger):
            init_loggers.add(log)
        else:
            raise TypeError("`loggers` must be a list of str or tune "
                            "loggers.")

    return list(init_loggers)


class TuneBaseSearchCV(BaseEstimator):
    """Abstract base class for TuneGridSearchCV and TuneSearchCV"""

    defined_schedulers = [
        "PopulationBasedTraining", "AsyncHyperBandScheduler",
        "HyperBandScheduler", "MedianStoppingRule", "ASHAScheduler",
        "HyperBandForBOHB"
    ]
    defined_loggers = ["tensorboard", "csv", "mlflow", "json"]

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
        self._check_is_fitted("best_params_", check_refit="multimetric")
        return self.best_params

    @property
    def best_index_(self):
        """int: The index (of the ``cv_results_`` arrays)
        which corresponds to the best candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

        """
        self._check_is_fitted("best_index_", check_refit="multimetric")
        return self.best_index

    @property
    def best_score_(self):
        """float: Mean cross-validated score of the best_estimator

        For multi-metric evaluation, this is present only if ``refit``
        is specified.

        """
        self._check_is_fitted("best_score_", check_refit="multimetric")
        return self.best_score

    @property
    def multimetric_(self):
        """bool: Whether evaluation performed was multi-metric."""
        return self.is_multi

    @property
    def classes_(self):
        """list: Get the list of unique classes found in the target `y`."""
        self._check_is_fitted("classes_")
        return self.best_estimator_.classes_

    @property
    def n_splits_(self):
        """int: The number of cross-validation splits (folds/iterations)."""
        self._check_is_fitted("n_splits_", check_refit=False)
        return self.n_splits_

    @property
    def best_estimator_(self):
        """estimator: Estimator that was chosen by the search,
        i.e. estimator which gave highest score (or smallest loss if
        specified) on the left out data. Not available if ``refit=False``.

        See ``refit`` parameter for more information on allowed values.
        """
        self._check_is_fitted("best_estimator_")
        return self.best_estimator

    @property
    def refit_time_(self):
        """float: Seconds used for refitting the best model on the
        whole dataset.

        This is present only if ``refit`` is not False.
        """
        self._check_is_fitted("refit_time_")
        return self.refit_time

    @property
    def scorer_(self):
        """function or a dict: Scorer function used on the held out
        data to choose the best parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.
        """
        return self.scoring

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

    def _check_is_fitted(self, method_name, check_refit=True):
        """Helper method to see if the estimator has been fitted.

        Args:
            method_name (str): String of the method name called from the user.

            check_refit (bool|str): Whether to also check for `self.refit`
                param. If "multimetric", will only check if `self.multimetric`
                param is also True. Defaults to True.

        Raises:
            NotFittedError: If the estimator has not been fitted.
            TypeError: If the estimator is invalid (i.e. doesn't implement
                the sklearn estimator interface).

        """
        if not self.refit:
            if check_refit == "multimetric":
                if self.is_multi:
                    msg = (
                        "This {0} instance was initialized with refit=False. "
                        "For multi-metric evaluation, {1} "
                        "is available only after refitting on the best "
                        "parameters.").format(
                            type(self).__name__, method_name)
                    raise NotFittedError(msg)
            elif check_refit:
                msg = (
                    "This {0} instance was initialized with refit=False. {1} "
                    "is available only after refitting on the best "
                    "parameters.").format(type(self).__name__, method_name)
                raise NotFittedError(msg)
        check_is_fitted(self)

    def _is_multimetric(self, scoring):
        """Helper method to see if multimetric scoring is
        being used

        Args:
            scoring (str, callable, list, tuple, or dict):
                the scoring being used
        """

        return isinstance(scoring, (list, tuple, dict))

    def __init__(self,
                 estimator,
                 early_stopping=None,
                 scoring=None,
                 n_jobs=None,
                 cv=5,
                 refit=True,
                 verbose=0,
                 error_score="raise",
                 return_train_score=False,
                 local_dir="~/ray_results",
                 max_iters=1,
                 use_gpu=False,
                 loggers=None,
                 pipeline_auto_early_stop=True,
                 stopper=None,
                 time_budget_s=None):
        if max_iters < 1:
            raise ValueError("max_iters must be greater than or equal to 1.")
        self.estimator = estimator
        self.base_estimator = estimator
        self.pipeline_auto_early_stop = pipeline_auto_early_stop
        self.stopper = stopper
        self.time_budget_s = time_budget_s

        if self.pipeline_auto_early_stop and check_is_pipeline(estimator):
            _, self.base_estimator = self.base_estimator.steps[-1]

        self.early_stop_type = get_early_stop_type(self.base_estimator,
                                                   bool(early_stopping))

        if not self._can_early_stop():
            if early_stopping:
                raise ValueError("Early stopping is not supported because "
                                 "the estimator does not have `partial_fit`, "
                                 "does not support warm_start, or is a "
                                 "tree classifier. Set "
                                 "`early_stopping=False`.")
        if not early_stopping and max_iters > 1:
            warnings.warn(
                "max_iters is set > 1 but incremental/partial training "
                "is not enabled. To enable partial training, "
                "ensure the estimator has `partial_fit` or "
                "`warm_start` and set `early_stopping=True`. "
                "Automatically setting max_iters=1.",
                category=UserWarning)
            max_iters = 1

        # Get metric scoring name
        self.scoring = scoring
        self.refit = refit
        if not hasattr(self, "is_multi"):
            self.scoring, self.is_multi = _check_multimetric_scoring(
                self.estimator, self.scoring)

        if self.is_multi:
            self._base_metric_name = self.refit
        else:
            self._base_metric_name = "score"

        self._metric_name = "average_test_%s" % self._base_metric_name

        if early_stopping:
            if not self._can_early_stop() and is_lightgbm_model(
                    self.base_estimator):
                warnings.warn("lightgbm>=3.0.0 required for early_stopping "
                              "functionality.")
            assert self._can_early_stop()
            if max_iters == 1:
                warnings.warn(
                    "early_stopping is enabled but max_iters = 1. "
                    "To enable partial training, set max_iters > 1.",
                    category=UserWarning)
            if self.early_stop_type == EarlyStopping.XGB:
                warnings.warn(
                    "tune-sklearn implements incremental learning "
                    "for xgboost models following this: "
                    "https://github.com/dmlc/xgboost/issues/1686. "
                    "This may negatively impact performance. To "
                    "disable, set `early_stopping=False`.",
                    category=UserWarning)
            elif self.early_stop_type == EarlyStopping.LGBM:
                warnings.warn(
                    "tune-sklearn implements incremental learning "
                    "for lightgbm models following this: "
                    "https://lightgbm.readthedocs.io/en/latest/pythonapi/"
                    "lightgbm.LGBMModel.html#lightgbm.LGBMModel.fit "
                    "This may negatively impact performance. To "
                    "disable, set `early_stopping=False`.",
                    category=UserWarning)
            elif self.early_stop_type == EarlyStopping.CATBOOST:
                warnings.warn(
                    "tune-sklearn implements incremental learning "
                    "for Catboost models following this: "
                    "https://catboost.ai/docs/concepts/python-usages-"
                    "examples.html#training-continuation "
                    "This may negatively impact performance. To "
                    "disable, set `early_stopping=False`.",
                    category=UserWarning)
            if early_stopping is True:
                # Override the early_stopping variable so
                # that it is resolved appropriately in
                # the next block
                early_stopping = "AsyncHyperBandScheduler"
            # Resolve the early stopping object
            early_stopping = resolve_early_stopping(early_stopping, max_iters,
                                                    self._metric_name)

        self.early_stopping = early_stopping
        self.max_iters = max_iters

        self.cv = cv
        self.n_jobs = int(n_jobs or -1)
        self.sk_n_jobs = 1
        if os.environ.get("SKLEARN_N_JOBS") is not None:
            self.sk_n_jobs = int(os.environ.get("SKLEARN_N_JOBS"))

        self.verbose = verbose
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.local_dir = local_dir
        self.use_gpu = use_gpu
        self.loggers = resolve_loggers(loggers)
        assert isinstance(self.n_jobs, int)

    def _fit(self, X, y=None, groups=None, **fit_params):
        """Helper method to run fit procedure

        Args:
            X (:obj:`array-like` (shape = [n_samples, n_features])):
                Training vector, where n_samples is the number of samples and
                n_features is the number of features.
            y (:obj:`array-like`): Shape of array expected to be [n_samples]
                or [n_samples, n_output]). Target relative to X for
                classification or regression; None for unsupervised learning.
            groups (:obj:`array-like` (shape (n_samples,)), optional):
                Group labels for the samples used while splitting the dataset
                into train/test set. Only used in conjunction with a "Group"
                `cv` instance (e.g., `GroupKFold`).
            **fit_params (:obj:`dict` of str): Parameters passed to
                the ``fit`` method of the estimator.

        Returns:
            :obj:`TuneBaseSearchCV` child instance, after fitting.
        """

        self._check_params()
        classifier = is_classifier(self.estimator)
        cv = check_cv(cv=self.cv, y=y, classifier=classifier)
        self.n_splits = cv.get_n_splits(X, y, groups)
        if not hasattr(self, "is_multi"):
            self.scoring, self.is_multi = _check_multimetric_scoring(
                self.estimator, self.scoring)
        else:
            self.scoring, _ = _check_multimetric_scoring(
                self.estimator, self.scoring)

        if self.is_multi:
            if self.refit and (not isinstance(self.refit, str)
                               or self.refit not in self.scoring):
                raise ValueError("When using multimetric scoring, refit "
                                 "must be the name of the scorer used to "
                                 "pick the best parameters. If not needed, "
                                 "set refit to False")

        assert isinstance(
            self.n_jobs,
            int), ("Internal error: self.n_jobs must be an integer.")
        if self.n_jobs < 0:
            resources_per_trial = {"cpu": 1, "gpu": 1 if self.use_gpu else 0}
            if self.n_jobs < -1:
                warnings.warn(
                    "`self.n_jobs` is automatically set "
                    "-1 for any negative values.",
                    category=UserWarning)
        else:
            available_cpus = multiprocessing.cpu_count()
            gpu_fraction = 1 if self.use_gpu else 0
            if ray.is_initialized():
                available_cpus = ray.cluster_resources()["CPU"]
                if self.use_gpu:
                    available_gpus = ray.cluster_resources()["GPU"]
                    gpu_fraction = available_gpus / self.n_jobs
            cpu_fraction = available_cpus / self.n_jobs
            if cpu_fraction > 1:
                cpu_fraction = int(np.ceil(cpu_fraction))
            if gpu_fraction > 1:
                gpu_fraction = int(np.ceil(gpu_fraction))
            resources_per_trial = {"cpu": cpu_fraction, "gpu": gpu_fraction}

        X_id = ray.put(X)
        y_id = ray.put(y)

        config = {}
        config["early_stopping"] = bool(self.early_stopping)
        config["early_stop_type"] = self.early_stop_type
        config["X_id"] = X_id
        config["y_id"] = y_id
        config["groups"] = groups
        config["cv"] = cv
        config["fit_params"] = fit_params
        config["scoring"] = self.scoring
        config["max_iters"] = self.max_iters
        config["return_train_score"] = self.return_train_score
        config["n_jobs"] = self.sk_n_jobs
        config["metric_name"] = self._metric_name

        self._fill_config_hyperparam(config)
        analysis = self._tune_run(config, resources_per_trial)

        self.cv_results_ = self._format_results(self.n_splits, analysis)

        metric = self._metric_name
        base_metric = self._base_metric_name

        # For multi-metric evaluation, store the best_index, best_params and
        # best_score iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.is_multi:
            # If callable, refit is expected to return the index of the best
            # parameter set.
            if callable(self.refit):
                self.best_index = self.refit(self.cv_results_)
                if not isinstance(self.best_index, numbers.Integral):
                    raise TypeError("best_index returned is not an integer")
                if (self.best_index < 0
                        or self.best_index >= len(self.cv_results_["params"])):
                    raise IndexError("best_index index out of range")
            else:
                self.best_index = self.cv_results_["rank_test_%s" %
                                                   base_metric].argmin()
                self.best_score = self.cv_results_[
                    "mean_test_%s" % base_metric][self.best_index]
            best_config = analysis.get_best_config(
                metric=metric, mode="max", scope="last")
            self.best_params = self._clean_config_dict(best_config)

        if self.refit:
            base_estimator = clone(self.estimator)
            if self.early_stop_type == EarlyStopping.WARM_START_ENSEMBLE:
                logger.info("tune-sklearn uses `n_estimators` to warm "
                            "start, so this parameter can't be "
                            "set when warm start early stopping. "
                            "`n_estimators` defaults to `max_iters`.")
                if check_is_pipeline(base_estimator):
                    cloned_final_estimator = base_estimator.steps[-1][1]
                    cloned_final_estimator.set_params(
                        **{"n_estimators": self.max_iters})
                else:
                    self.best_params["n_estimators"] = self.max_iters
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator = clone(
                base_estimator.set_params(**self.best_params))
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator.fit(X, y, **fit_params)
            else:
                self.best_estimator.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time = refit_end_time - refit_start_time

        return self

    def fit(self, X, y=None, groups=None, **fit_params):
        """Run fit with all sets of parameters.

        ``tune.run`` is used to perform the fit procedure.

        Args:
            X (:obj:`array-like` (shape = [n_samples, n_features])):
                Training vector, where n_samples is the number of samples and
                n_features is the number of features.
            y (:obj:`array-like`): Shape of array expected to be [n_samples]
                or [n_samples, n_output]). Target relative to X for
                classification or regression; None for unsupervised learning.
            groups (:obj:`array-like` (shape (n_samples,)), optional):
                Group labels for the samples used while splitting the dataset
                into train/test set. Only used in conjunction with a "Group"
                `cv` instance (e.g., `GroupKFold`).
            **fit_params (:obj:`dict` of str): Parameters passed to
                the ``fit`` method of the estimator.

        Returns:
            :obj:`TuneBaseSearchCV` child instance, after fitting.

        """
        ray_init = ray.is_initialized()
        try:
            if not ray_init:
                if self.n_jobs == 1:
                    ray.init(
                        local_mode=True,
                        configure_logging=False,
                        ignore_reinit_error=True,
                        include_dashboard=False)
                else:
                    ray.init(
                        ignore_reinit_error=True,
                        configure_logging=False,
                        include_dashboard=False
                        # log_to_driver=self.verbose == 2
                    )
                    if self.verbose != 2:
                        logger.info("TIP: Hiding process output by default. "
                                    "To show process output, set verbose=2.")

            result = self._fit(X, y, groups, **fit_params)

            if not ray_init and ray.is_initialized():
                ray.shutdown()

            return result

        except Exception:
            if not ray_init and ray.is_initialized():
                ray.shutdown()
            raise

    def score(self, X, y=None):
        """Compute the score(s) of an estimator on a given test set.

        Args:
            X (:obj:`array-like` (shape = [n_samples, n_features])): Input
                data, where n_samples is the number of samples and
                n_features is the number of features.
            y (:obj:`array-like`): Shape of array is expected to be
                [n_samples] or [n_samples, n_output]). Target relative to X
                for classification or regression. You can also pass in
                None for unsupervised learning.

        Returns:
            float: computed score

        """
        self._check_is_fitted(self._metric_name)
        if self.is_multi:
            scorer_name = self.refit
        else:
            scorer_name = "score"
        return self.scoring[scorer_name](self.best_estimator_, X, y)

    def _can_early_stop(self):
        """Helper method to determine if it is possible to do early stopping.

        Only sklearn estimators with `partial_fit` or `warm_start` can be early
        stopped. warm_start works by picking up training from the previous
        call to `fit`.

        Returns:
            bool: if the estimator can early stop

        """
        return self.early_stop_type != EarlyStopping.NO_EARLY_STOP

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
                "estimator_list", "early_stopping", "X_id", "y_id", "groups",
                "cv", "fit_params", "scoring", "max_iters",
                "return_train_score", "n_jobs", "metric_name",
                "early_stop_type"
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
        trials = [
            trial for trial in out.trials if trial.status == Trial.TERMINATED
        ]
        trial_dirs = [trial.logdir for trial in trials]
        # The result dtaframes are indexed by their trial logdir
        trial_dfs = out.fetch_trial_dataframes()

        # Try to find a template df to use for trials that did not return
        # any results. These trials should copy the structure and fill it
        # with NaNs so that the later reshape actions work.
        template_df = None
        fix_trial_dirs = []  # Holds trial dirs with no results
        for trial_dir in trial_dirs:
            if trial_dir in trial_dfs and template_df is None:
                template_df = trial_dfs[trial_dir]
            elif trial_dir not in trial_dfs:
                fix_trial_dirs.append(trial_dir)

        # Create NaN dataframes for trials without results
        if fix_trial_dirs:
            if template_df is None:
                # No trial returned any results
                return {}
            for trial_dir in fix_trial_dirs:
                trial_df = pd.DataFrame().reindex_like(template_df)
                trial_dfs[trial_dir] = trial_df

        # Keep right order
        dfs = [trial_dfs[trial_dir] for trial_dir in trial_dirs]
        finished = [df.iloc[[-1]] for df in dfs]
        test_scores = {}
        train_scores = {}
        for name in self.scoring:
            test_scores[name] = [
                df[[
                    col for col in dfs[0].columns
                    if "split" in col and "test_%s" % name in col
                ]].to_numpy() for df in finished
            ]
            if self.return_train_score:
                train_scores[name] = [
                    df[[
                        col for col in dfs[0].columns
                        if "split" in col and "train_%s" % name in col
                    ]].to_numpy() for df in finished
                ]
            else:
                train_scores = None

        configs = [trial.config for trial in trials]
        candidate_params = [
            self._clean_config_dict(config) for config in configs
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

        for name in self.scoring:
            _store(
                results,
                "test_%s" % name,
                test_scores[name],
                n_splits,
                n_candidates,
                splits=True,
                rank=True,
            )
        if self.return_train_score:
            for name in self.scoring:
                _store(
                    results,
                    "train_%s" % name,
                    train_scores[name],
                    n_splits,
                    n_candidates,
                    splits=True,
                    rank=True,
                )

        results["time_total_s"] = np.array(
            [df["time_total_s"].to_numpy() for df in finished]).flatten()

        results["training_iteration"] = np.array([
            df["training_iteration"].to_numpy() for df in finished
        ]).flatten()

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
