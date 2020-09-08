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
import os


class TuneGridSearchCV(TuneBaseSearchCV):
    """Exhaustive search over specified parameter values for an estimator.

    Important members are fit, predict.

    GridSearchCV implements a "fit" and a "score" method. It also implements
    "predict", "predict_proba", "decision_function", "transform" and
    "inverse_transform" if they are implemented in the estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid.

    Args:
        estimator (`estimator`): Object that implements the
            scikit-learn estimator interface. Either estimator needs to
            provide a ``score`` function, or ``scoring`` must be passed.
        param_grid (`dict` or `list` of `dict`): Dictionary with parameters
            names (string) as keys and lists of
            parameter settings to try as values, or a list of such
            dictionaries, in which case the grids spanned by each dictionary
            in the list are explored. This enables searching over any sequence
            of parameter settings.
        early_stopping (bool, str or :class:`TrialScheduler`, optional): Option
            to stop fitting to a hyperparameter configuration if it performs
            poorly. Possible inputs are:

            - If True, defaults to ASHAScheduler.
            - A string corresponding to the name of a Tune Trial Scheduler
              (i.e., "ASHAScheduler"). To specify parameters of the scheduler,
              pass in a scheduler object instead of a string.
            - Scheduler for executing fit with early stopping. Only a subset
              of schedulers are currently supported. The scheduler will only be
              used if the estimator supports partial fitting
            - If None or False, early stopping will not be used.

        scoring (str, list/tuple, dict, or None): A single
            string or a callable to evaluate the predictions on the test set.
            See https://scikit-learn.org/stable/modules/model_evaluation.html
            #scoring-parameter for all options.
            For evaluating multiple metrics, either give a list/tuple of
            (unique) strings or a dict with names as keys and callables as
            values.
            If None, the estimator's score method is used. Defaults to None.
        n_jobs (int): Number of jobs to run in parallel. None or -1 means
            using all processors. Defaults to None. If set to 1, jobs
            will be run using Ray's 'local mode'. This can
            lead to significant speedups if the model takes < 10 seconds
            to fit due to removing inter-process communication overheads.
        sk_n_jobs (int): Number of jobs to run in parallel for cross validating
            each hyperparameter set; the ``n_jobs`` parameter for
            ``cross_validate`` call to sklearn when early stopping isn't used.
        cv (int, `cross-validation generator` or `iterable`): Determines the
            cross-validation splitting strategy. Possible inputs for cv are:

            - None, to use the default 5-fold cross validation,
            - integer, to specify the number of folds in a `(Stratified)KFold`,
            - An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if the estimator is a classifier and ``y``
            is either binary or multiclass, :class:`StratifiedKFold` is used.
            In all other cases, :class:`KFold` is used. Defaults to None.
        refit (bool or str):
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
            w.r.t this specific scorer. If refit not needed, set to False.
            See ``scoring`` parameter to know more about multiple metric
            evaluation.
            Defaults to True.
        verbose (int): Controls the verbosity: 0 = silent, 1 = only
            status updates, 2 = status and trial results. Defaults to 0.
        error_score ('raise' or int or float): Value to assign to the score
            if an error occurs in estimator fitting. If set to 'raise',
            the error is raised. If a numeric value
            is given, FitFailedWarning is raised. This parameter does not
            affect the refit step, which will always raise the error.
            Defaults to np.nan.
        return_train_score (bool): If ``False``, the ``cv_results_``
            attribute will not include training scores. Defaults to False.
            Computing training scores is used to get insights on how different
            parameter settings impact the overfitting/underfitting trade-off.
            However computing the scores on the training set can be
            computationally expensive and is not strictly required to select
            the parameters that yield the best generalization performance.
        local_dir (str): A string that defines where checkpoints will
            be stored. Defaults to "~/ray_results"
        max_iters (int): Indicates the maximum number of epochs to run for each
            hyperparameter configuration sampled.
            This parameter is used for early stopping. Defaults to 1.
            When using warm start to early stop on ensembles, this will
            determine `n_estimators` for the final refitted ensemble.`
        use_gpu (bool): Indicates whether to use gpu for fitting.
            Defaults to False. If True, training will use 1 gpu
            for `resources_per_trial`.
    """

    def __init__(self,
                 estimator,
                 param_grid,
                 early_stopping=None,
                 scoring=None,
                 n_jobs=None,
                 sk_n_jobs=-1,
                 cv=5,
                 refit=True,
                 verbose=0,
                 error_score="raise",
                 return_train_score=False,
                 local_dir="~/ray_results",
                 max_iters=1,
                 use_gpu=False):
        super(TuneGridSearchCV, self).__init__(
            estimator=estimator,
            early_stopping=early_stopping,
            scoring=scoring,
            n_jobs=n_jobs or -1,
            sk_n_jobs=sk_n_jobs,
            cv=cv,
            refit=refit,
            error_score=error_score,
            return_train_score=return_train_score,
            local_dir=local_dir,
            max_iters=max_iters,
            verbose=verbose,
            use_gpu=use_gpu)

        _check_param_grid(param_grid)
        self.param_grid = param_grid

    def _fill_config_hyperparam(self, config):
        """Fill in the ``config`` dictionary with the hyperparameters.

        Each distribution is converted to a list, then returns a
        dictionary showing the values of the hyperparameters that
        have been grid searched over.

        Args:
            config (`dict`): dictionary to be filled in as the
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
            analysis (`ExperimentAnalysis`): Object returned by
                `tune.run`.

        """
        if self.early_stopping is not None:
            config["estimator_list"] = [
                clone(self.estimator) for _ in range(self.n_splits)
            ]
        else:
            config["estimator_list"] = [self.estimator]

        if isinstance(self.param_grid, list):
            analysis = tune.run(
                _Trainable,
                search_alg=ListSearcher(self.param_grid),
                num_samples=self._list_grid_num_samples(),
                scheduler=self.early_stopping,
                reuse_actors=True,
                verbose=self.verbose,
                stop={"training_iteration": self.max_iters},
                config=config,
                fail_fast=True,
                resources_per_trial=resources_per_trial,
                local_dir=os.path.expanduser(self.local_dir))
        else:
            analysis = tune.run(
                _Trainable,
                scheduler=self.early_stopping,
                reuse_actors=True,
                verbose=self.verbose,
                stop={"training_iteration": self.max_iters},
                config=config,
                fail_fast=True,
                resources_per_trial=resources_per_trial,
                local_dir=os.path.expanduser(self.local_dir))

        return analysis
