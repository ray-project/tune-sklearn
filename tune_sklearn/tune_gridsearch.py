"""Class for doing grid search over lists of hyperparameters
    -- Anthony Yu and Michael Chau
"""

from tune_sklearn.tune_basesearch import TuneBaseSearchCV
from tune_sklearn._trainable import _Trainable
from sklearn.model_selection._search import _check_param_grid
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from ray import tune
from tune_sklearn.list_searcher import ListSearcher


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

        max_iters (int):
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
            early_stopping=False,
            max_iters=10,
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
            max_iters=max_iters,
        )

        _check_param_grid(param_grid)
        self.param_grid = param_grid

    def _fill_config_hyperparam(self, config):
        """Fill in the ``config`` dictionary with the hyperparameters.

        Each distribution is converted to a list, then returns a
        dictionary showing the values of the hyperparameters that
        have been grid searched over.

        Args:
            config (:obj:`dict`): dictionary to be filled in as the
                configuration for `tune.run`.

        """
        if isinstance(self.param_grid, list):
            return

        for key, distribution in self.param_grid.items():
            config[key] = tune.grid_search(list(distribution))

    def _list_grid_num_samples(self):
        """Calculate the num_samples for `tune.run`.

        This is used when a list of dictionaries is passed in
        for the `param_grid`
        """
        return len(list(ParameterGrid(self.param_grid)))

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
        else:
            config["estimator"] = self.estimator

        if isinstance(self.param_grid, list):
            analysis = tune.run(
                _Trainable,
                search_alg=ListSearcher(self.param_grid),
                num_samples=self._list_grid_num_samples(),
                scheduler=self.scheduler,
                reuse_actors=True,
                verbose=self.verbose,
                stop={"training_iteration": self.max_iters},
                config=config,
                checkpoint_at_end=True,
                resources_per_trial=resources_per_trial,
            )
        else:
            analysis = tune.run(
                _Trainable,
                scheduler=self.scheduler,
                reuse_actors=True,
                verbose=self.verbose,
                stop={"training_iteration": self.max_iters},
                config=config,
                checkpoint_at_end=True,
                resources_per_trial=resources_per_trial,
            )

        return analysis
