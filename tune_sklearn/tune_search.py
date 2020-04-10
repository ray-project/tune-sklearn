"""A GridSearchCV interface built with a Ray Tune back-end.

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
from sklearn.model_selection import (
    cross_validate,
    check_cv,
    ParameterGrid,
    ParameterSampler,
)
from sklearn.model_selection._search import _check_param_grid
from sklearn.metrics import check_scoring
from sklearn.base import is_classifier
from sklearn.utils.metaestimators import _safe_split
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
import ray
from ray import tune
from ray.tune import Trainable
from ray.tune.schedulers import (
    PopulationBasedTraining, AsyncHyperBandScheduler, HyperBandScheduler,
    HyperBandForBOHB, MedianStoppingRule, TrialScheduler)
import numpy as np
from numpy.ma import MaskedArray
import os
from pickle import PicklingError
import cloudpickle as cpickle
import warnings


# Helper class to train models
class _Trainable(Trainable):
    """Class to be passed in as the first argument of tune.run to train models.

    Overrides Ray Tune's Trainable class to specify the setup, train, save,
    and restore routines.

    """

    def _setup(self, config):
        """Sets up Trainable attributes during initialization.

        Also sets up parameters for the sklearn estimator passed in.

        Args:
            config (dict): contains necessary parameters to complete the `fit`
                routine for the estimator. Also includes parameters for early
                stopping if it is set to true.

        """
        self.estimator = clone(config.pop("estimator"))
        self.scheduler = config.pop("scheduler")
        X_id = config.pop("X_id")
        self.X = ray.get(X_id)

        y_id = config.pop("y_id")
        self.y = ray.get(y_id)
        self.groups = config.pop("groups")
        self.fit_params = config.pop("fit_params")
        self.scoring = config.pop("scoring")
        self.early_stopping = config.pop("early_stopping")
        self.early_stopping_max_epochs = config.pop(
            "early_stopping_max_epochs")
        self.cv = config.pop("cv")
        self.return_train_score = config.pop("return_train_score")

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
        """Trains one iteration of the model called when ``tune.run`` is called.

        Different routines are run depending on if the ``early_stopping``
        attribute is True or not.

        If ``self.early_stopping`` is True, each fold is fit with
        `partial_fit`, which stops training the model if the validation
        score is not improving for a particular fold.

        Otherwise, run the full cross-validation procedure.

        In both cases, the average test accuracy is returned over all folds,
        as well as the individual folds' accuracies as a dictionary.

        Returns:
            ret (:obj:`dict): Dictionary of results as a basis for
                ``cv_results_`` for one of the cross-validation interfaces.

        """
        if self.early_stopping:
            for i, (train, test) in enumerate(self.cv.split(self.X, self.y)):
                X_train, y_train = _safe_split(self.estimator[i], self.X,
                                               self.y, train)
                X_test, y_test = _safe_split(
                    self.estimator[i],
                    self.X,
                    self.y,
                    test,
                    train_indices=train)
                self.estimator[i].partial_fit(X_train, y_train,
                                              np.unique(self.y))
                if self.return_train_score:
                    self.fold_train_scores[i] = self.scoring(
                        self.estimator[i], X_train, y_train)
                self.fold_scores[i] = self.scoring(self.estimator[i], X_test,
                                                   y_test)

            ret = {}
            total = 0
            for i, score in enumerate(self.fold_scores):
                total += score
                key_str = f"split{i}_test_score"
                ret[key_str] = score
            self.mean_score = total / len(self.fold_scores)
            ret["average_test_score"] = self.mean_score

            if self.return_train_score:
                total = 0
                for i, score in enumerate(self.fold_train_scores):
                    total += score
                    key_str = f"split{i}_train_score"
                    ret[key_str] = score
                self.mean_train_score = total / len(self.fold_train_scores)
                ret["average_train_score"] = self.mean_train_score

            return ret
        else:
            scores = cross_validate(
                self.estimator,
                self.X,
                self.y,
                cv=self.cv,
                fit_params=self.fit_params,
                groups=self.groups,
                scoring=self.scoring,
                return_train_score=self.return_train_score,
            )

            ret = {}
            for i, score in enumerate(scores["test_score"]):
                key_str = f"split{i}_test_score"
                ret[key_str] = score
            self.test_accuracy = sum(scores["test_score"]) / len(
                scores["test_score"])
            ret["average_test_score"] = self.test_accuracy

            if self.return_train_score:
                for i, score in enumerate(scores["train_score"]):
                    key_str = f"split{i}_train_score"
                    ret[key_str] = score
                self.train_accuracy = sum(scores["train_score"]) / len(
                    scores["train_score"])
                ret["average_train_score"] = self.train_accuracy

            return ret

    def _save(self, checkpoint_dir):
        """Creates a checkpoint in ``checkpoint_dir``, creating a pickle file.

        Args:
            checkpoint_dir (str): file path to store pickle checkpoint.

        Returns:
            path (str): file path to the pickled checkpoint file.

        """
        path = os.path.join(checkpoint_dir, "checkpoint")
        with open(path, "wb") as f:
            try:
                cpickle.dump(self.estimator, f)
                self.pickled = True
            except PicklingError:
                self.pickled = False
                warnings.warn("{} could not be pickled. "
                              "Restoring estimators may run into issues."
                              .format(self.estimator))
        return path

    def _restore(self, checkpoint):
        """Loads a checkpoint created from `_save`.

        Args:
            checkpoint (str): file path to pickled checkpoint file.

        """
        if self.pickled:
            with open(checkpoint, "rb") as f:
                self.estimator = cpickle.load(f)
        else:
            warnings.warn("No estimator restored")

    def reset_config(self, new_config):
        return False


class TuneBaseSearchCV(BaseEstimator):
    """Abstract base class for TuneGridSearchCV and TuneRandomizedSearchCV"""

    defined_schedulers = [
        "PopulationBasedTraining", "AsyncHyperBandScheduler",
        "HyperBandScheduler", "HyperBandForBOHB", "MedianStoppingRule"
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
            early_stopping_max_epochs=10,
    ):
        self.estimator = estimator
        self.early_stopping = self._can_early_stop()
        if self.early_stopping:
            self.early_stopping_max_epochs = early_stopping_max_epochs
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
                else:
                    raise ValueError("{} is not a defined scheduler. "
                                     "Check the list of available schedulers."
                                     .format(scheduler))
            elif isinstance(scheduler, TrialScheduler) or scheduler is None:
                self.scheduler = scheduler
                if self.scheduler is not None:
                    self.scheduler.metric = "average_test_score"
            else:
                raise TypeError("Scheduler must be a str or tune scheduler")
        else:
            warnings.warn("Unable to do early stopping because "
                          "estimator does not have `partial_fit`")
            self.early_stopping_max_epochs = 1
            self.scheduler = None
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.verbose = verbose
        self.error_score = error_score
        self.return_train_score = return_train_score

    def _get_param_iterator(self):
        """Get a parameter iterator to be passed in to _format_results to
        generate the cv_results_ dictionary.

        Method should be overridden in a child class to generate different
        iterators depending on whether it is randomized or grid search.

        """
        raise NotImplementedError("Implement in a child class.")

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
        ray.init(ignore_reinit_error=True)

        self._check_params()
        classifier = is_classifier(self.estimator)
        cv = check_cv(self.cv, y, classifier)
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
        config["early_stopping_max_epochs"] = self.early_stopping_max_epochs
        config["return_train_score"] = self.return_train_score

        candidate_params = list(self._get_param_iterator())

        self._fill_config_hyperparam(config)
        analysis = self._tune_run(config, resources_per_trial)

        self.cv_results_ = self._format_results(candidate_params,
                                                self.n_splits, analysis)

        if self.refit:
            best_config = analysis.get_best_config(
                metric="average_test_score", mode="max")
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
                    "early_stopping_max_epochs",
                    "return_train_score",
            ]:
                best_config.pop(key)
            self.best_params = best_config
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

    def _format_results(self, candidate_params, n_splits, out):
        """Helper to generate the ``cv_results_`` dictionary.

        Args:
            candidate_params (:obj:`list` of :obj:`dict`): List of all
                hyperparameterconfigurations encoded as a dictionary, where the
                keys are the hyperparameter variables and the values are the
                numeric values set to those variables.
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


class TuneRandomizedSearchCV(TuneBaseSearchCV):
    """Randomized search on hyper parameters.

    RandomizedSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.

    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by n_iter.

    Args:
        estimator (:obj:`estimator`): This is assumed to implement the
            scikit-learn estimator interface.
            Either estimator needs to provide a ``score`` function,
            or ``scoring`` must be passed.

        param_distributions (:obj:`dict`):
            Dictionary with parameters names (string) as keys and distributions
            or lists of parameter settings to try.

            Distributions must provide a rvs  method for sampling (such as
            those from scipy.stats.distributions).

            If a list is given, it is sampled uniformly. If a list of dicts is
            given, first a dict is sampled uniformly, and then a parameter is
            sampled using that dict as above.

        scheduler (str or :obj:`TrialScheduler`, optional):
            Scheduler for executing fit. Refer to ray.tune.schedulers for all
            options. If a string is given, a scheduler will be created with
            default parameters. To specify parameters of the scheduler, pass in
            a scheduler object instead of a string. The scheduler will be
            used if the estimator supports partial fitting to stop fitting to a
            hyperparameter configuration if it performs poorly.

            If None, the FIFO scheduler will be used. Defaults to None.

        n_iter (int):
            Number of parameter settings that are sampled. n_iter trades
            off runtime vs quality of the solution. Defaults to 10.

        scoring (str, :obj:`callable`, :obj:`list`, :obj:`tuple`, :obj:`dict`
            or None):
            A single string (see :ref:`scoring_parameter`) or a callable
            (see :ref:`scoring`) to evaluate the predictions on the test set.

            For evaluating multiple metrics, either give a list of (unique)
            strings or a dict with names as keys and callables as values.

            NOTE that when using custom scorers, each scorer should return a
            single value. Metric functions returning a list/array of values
            can be wrapped into multiple scorers that return one value each.

            If None, the estimator's score method is used. Defaults to None.

        n_jobs (int):
            Number of jobs to run in parallel. None or -1 means using all
            processors. Defaults to None.

        refit (bool, str, or :obj:`callable`):
            Refit an estimator using the best found parameters on the whole
            dataset.

            For multiple metric evaluation, this needs to be a string denoting
            the scorer that would be used to find the best parameters for
            refitting the estimator at the end.

            The refitted estimator is made available at the ``best_estimator_``
            attribute and permits using ``predict`` directly on this
            ``GridSearchCV`` instance.

            Also for multiple metric evaluation, the attributes
            ``best_index_``, ``best_score_`` and ``best_params_`` will only be
            available if ``refit`` is set and all of them will be determined
            w.r.t this specific scorer. ``best_score_`` is not returned if
            refit is callable.

            See ``scoring`` parameter to know more about multiple metric
            evaluation.

            Defaults to True.

        cv (int, :obj`cross-validation generator` or :obj:`iterable`):
            Determines the cross-validation splitting strategy.

            Possible inputs for cv are:
            - None, to use the default 5-fold cross validation,
            - integer, to specify the number of folds in a `(Stratified)KFold`,
            - An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if the estimator is a classifier and ``y``
            is either binary or multiclass, :class:`StratifiedKFold` is used.
            In all other cases, :class:`KFold` is used. Defaults to None.

        verbose (int):
            Controls the verbosity: 0 = silent, 1 = only status updates,
            2 = status and trial results. Defaults to 0.

        random_state (int or :obj:`RandomState`):
            Pseudo random number generator state used for random uniform
            sampling from lists of possible values instead of scipy.stats
            distributions.

            If int, random_state is the seed used by the random number
            generator;
            If RandomState instance, random_state is the random number
            generator;
            If None, the random number generator is the RandomState instance
            used by np.random. Defaults to None.

        error_score ('raise' or int or float):
            Value to assign to the score if an error occurs in estimator
            fitting. If set to 'raise', the error is raised. If a numeric value
            is given, FitFailedWarning is raised. This parameter does not
            affect the refit step, which will always raise the error.
            Defaults to np.nan.

        return_train_score (bool):
            If ``False``, the ``cv_results_`` attribute will not include
            training scores. Defaults to False.

            Computing training scores is used to get insights on how different
            parameter settings impact the overfitting/underfitting trade-off.

            However computing the scores on the training set can be
            computationally expensive and is not strictly required to select
            the parameters that yield the best generalization performance.

        early_stopping_max_epochs (int):
            Indicates the maximum number of epochs to run for each
            hyperparameter configuration sampled (specified by ``n_iter``).
            This parameter is used for early stopping. Defaults to 10.

    """

    def __init__(
            self,
            estimator,
            param_distributions,
            scheduler=None,
            n_iter=10,
            scoring=None,
            n_jobs=None,
            refit=True,
            cv=None,
            verbose=0,
            random_state=None,
            error_score=np.nan,
            return_train_score=False,
            early_stopping_max_epochs=10,
    ):
        super(TuneRandomizedSearchCV, self).__init__(
            estimator=estimator,
            scheduler=scheduler,
            scoring=scoring,
            n_jobs=n_jobs,
            cv=cv,
            verbose=verbose,
            refit=refit,
            error_score=error_score,
            return_train_score=return_train_score,
            early_stopping_max_epochs=early_stopping_max_epochs,
        )

        self.param_distributions = param_distributions
        self.num_samples = n_iter
        self.random_state = random_state

    def _get_param_iterator(self):
        """Gets a ParameterSampler instance based on the hyperparameter
        distributions.

        Returns:
            :obj:`ParameterSampler` instance.

        """
        return ParameterSampler(
            self.param_distributions,
            self.num_samples,
            random_state=self.random_state)

    def _fill_config_hyperparam(self, config):
        """Fill in the ``config`` dictionary with the hyperparameters.

        Each distribution in ``self.param_distributions`` must implement
        the ``rvs`` method to generate a random variable. The [0] is
        present to extract the single value out of a list, which is returned
        by ``rvs``.

        Args:
            config (:obj:`dict`): dictionary to be filled in as the
                configuration for `tune.run`.

        """
        samples = 1
        all_lists = True
        for key, distribution in self.param_distributions.items():
            if isinstance(distribution, list):
                import random

                config[key] = tune.sample_from(
                    lambda spec: distribution[random.randint(
                        0, len(distribution) - 1)]
                )
                samples *= len(distribution)
            else:
                all_lists = False
                config[key] = tune.sample_from(
                    lambda spec: distribution.rvs(1)[0])
        if all_lists:
            self.num_samples = min(self.num_samples, samples)

    def _tune_run(self, config, resources_per_trial):
        """Wrapper to call ``tune.run``. Multiple estimators are generated when
        early stopping is possible, whereas a single estimator is
        generated when  early stopping is not possible.

        Args:
            config (dict): Configurations such as hyperparameters to run
            ``tune.run`` on.

        resources_per_trial (dict): Resources to use per trial within Ray.
            Accepted keys are `cpu`, `gpu` and custom resources, and values
            are integers specifying the number of each resource to use.

        Returns:
            analysis (:obj:`ExperimentAnalysis`): Object returned by
                `tune.run`.

        """
        if self.early_stopping:
            config["estimator"] = [
                clone(self.estimator) for _ in range(self.n_splits)
            ]
            analysis = tune.run(
                _Trainable,
                scheduler=self.scheduler,
                reuse_actors=True,
                verbose=self.verbose,
                stop={"training_iteration": self.early_stopping_max_epochs},
                num_samples=self.num_samples,
                config=config,
                checkpoint_at_end=True,
                resources_per_trial=resources_per_trial,
            )
        else:
            config["estimator"] = self.estimator
            analysis = tune.run(
                _Trainable,
                scheduler=self.scheduler,
                reuse_actors=True,
                verbose=self.verbose,
                stop={"training_iteration": self.early_stopping_max_epochs},
                num_samples=self.num_samples,
                config=config,
                checkpoint_at_end=True,
                resources_per_trial=resources_per_trial,
            )

        return analysis


class TuneGridSearchCV(TuneBaseSearchCV):
    """Exhaustive search over specified parameter values for an estimator.

    Important members are fit, predict.

    GridSearchCV implements a "fit" and a "score" method. It also implements
    "predict", "predict_proba", "decision_function", "transform" and
    "inverse_transform" if they are implemented in the estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid.

    Args:
        estimator (:obj:`estimator`): This is assumed to implement the
                scikit-learn estimator interface.
                Either estimator needs to provide a ``score`` function,
                or ``scoring`` must be passed.

        param_grid (:obj:`dict` or :obj:`list` of :obj:`dict`):
            Dictionary with parameters names (string) as keys and lists of
            parameter settings to try as values, or a list of such
            dictionaries, in which case the grids spanned by each dictionary
            in the list are explored. This enables searching over any sequence
            of parameter settings.

        scheduler (str or :obj:`TrialScheduler`, optional):
            Scheduler for executing fit. Refer to ray.tune.schedulers for all
            options. If a string is given, a scheduler will be created with
            default parameters. To specify parameters of the scheduler, pass in
            a scheduler object instead of a string. The scheduler will be
            used if the estimator supports partial fitting to stop fitting to a
            hyperparameter configuration if it performs poorly.

            If None, the FIFO scheduler will be used. Defaults to None.

        scoring (str, :obj:`callable`, :obj:`list`, :obj:`tuple`, :obj:`dict`
            or None):
            A single string (see :ref:`scoring_parameter`) or a callable
            (see :ref:`scoring`) to evaluate the predictions on the test set.

            For evaluating multiple metrics, either give a list of (unique)
            strings or a dict with names as keys and callables as values.

            NOTE that when using custom scorers, each scorer should return a
            single value. Metric functions returning a list/array of values can
            be wrapped into multiple scorers that return one value each.

            If None, the estimator's score method is used. Defaults to None.

        n_jobs (int):
            Number of jobs to run in parallel. None or -1 means using all
            processors. Defaults to None.

        cv (int, :obj`cross-validation generator` or :obj:`iterable`):
            Determines the cross-validation splitting strategy.

            Possible inputs for cv are:
            - None, to use the default 5-fold cross validation,
            - integer, to specify the number of folds in a `(Stratified)KFold`,
            - An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if the estimator is a classifier and ``y``
            is either binary or multiclass, :class:`StratifiedKFold` is used.
            In all other cases, :class:`KFold` is used. Defaults to None.

        refit (bool, str, or :obj:`callable`):
            Refit an estimator using the best found parameters on the whole
            dataset.

            For multiple metric evaluation, this needs to be a string denoting
            the scorer that would be used to find the best parameters for
            refitting the estimator at the end.

            The refitted estimator is made available at the ``best_estimator_``
            attribute and permits using ``predict`` directly on this
            ``GridSearchCV`` instance.

            Also for multiple metric evaluation, the attributes
            ``best_index_``, ``best_score_`` and ``best_params_`` will only be
            available if ``refit`` is set and all of them will be determined
            w.r.t this specific scorer. ``best_score_`` is not returned if
            refit is callable.

            See ``scoring`` parameter to know more about multiple metric
            evaluation.

            Defaults to True.

        verbose (int):
            Controls the verbosity: 0 = silent, 1 = only status updates,
            2 = status and trial results. Defaults to 0.

        error_score ('raise' or int or float):
            Value to assign to the score if an error occurs in estimator
            fitting. If set to 'raise', the error is raised. If a numeric value
            is given, FitFailedWarning is raised. This parameter does not
            affect the refit step, which will always raise the error.
            Defaults to np.nan.

        return_train_score (bool):
            If ``False``, the ``cv_results_`` attribute will not include
            training scores. Defaults to False.

            Computing training scores is used to get insights on how different
            parameter settings impact the overfitting/underfitting trade-off.

            However computing the scores on the training set can be
            computationally expensive and is not strictly required to select
            the parameters that yield the best generalization performance.

        early_stopping_max_epochs (int):
            Indicates the maximum number of epochs to run for each
            hyperparameter configuration sampled (specified by ``n_iter``).
            This parameter is used for early stopping. Defaults to 10.

    """

    def __init__(
            self,
            estimator,
            param_grid,
            scheduler=None,
            scoring=None,
            n_jobs=None,
            cv=5,
            refit=True,
            verbose=0,
            error_score="raise",
            return_train_score=False,
            early_stopping_max_epochs=10,
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
            early_stopping_max_epochs=early_stopping_max_epochs,
        )

        _check_param_grid(param_grid)
        self.param_grid = param_grid

    def _get_param_iterator(self):
        """Gets the hyperparameter grid.

        Returns:
            :obj:`ParameterGrid` instance.

        """
        return ParameterGrid(self.param_grid)

    def _fill_config_hyperparam(self, config):
        """Fill in the ``config`` dictionary with the hyperparameters.

        Each distribution is converted to a list, then returns a
        dictionary showing the values of the hyperparameters that
        have been grid searched over.

        Args:
            config (:obj:`dict`): dictionary to be filled in as the
                configuration for `tune.run`.

        """
        for key, distribution in self.param_grid.items():
            config[key] = tune.grid_search(list(distribution))

    def _tune_run(self, config, resources_per_trial):
        """Wrapper to call ``tune.run``. Multiple estimators are generated when
        early stopping is possible, whereas a single estimator is
        generated when  early stopping is not possible.

        Args:
            config (dict): Configurations such as hyperparameters to run
                ``tune.run`` on.

        resources_per_trial (dict): Resources to use per trial within Ray.
            Accepted keys are `cpu`, `gpu` and custom resources, and values
            are integers specifying the number of each resource to use.

        Returns:
            analysis (:obj:`ExperimentAnalysis`): Object returned by
                `tune.run`.

        """
        if self.early_stopping:
            config["estimator"] = [
                clone(self.estimator) for _ in range(self.n_splits)
            ]
            analysis = tune.run(
                _Trainable,
                scheduler=self.scheduler,
                reuse_actors=True,
                verbose=self.verbose,
                stop={"training_iteration": self.early_stopping_max_epochs},
                config=config,
                checkpoint_at_end=True,
                resources_per_trial=resources_per_trial,
            )
        else:
            config["estimator"] = self.estimator
            analysis = tune.run(
                _Trainable,
                scheduler=self.scheduler,
                reuse_actors=True,
                verbose=self.verbose,
                stop={"training_iteration": self.early_stopping_max_epochs},
                config=config,
                checkpoint_at_end=True,
                resources_per_trial=resources_per_trial,
            )

        return analysis
