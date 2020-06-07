"""Class for cross-validation over distributions of hyperparameters
    -- Anthony Yu and Michael Chau
"""

from tune_sklearn.tune_basesearch import TuneBaseSearchCV
from tune_sklearn._trainable import _Trainable
from sklearn.base import clone
from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch
from tune_sklearn.list_searcher import RandomListSearcher
import numpy as np
import warnings


class TuneSearchCV(TuneBaseSearchCV):
    """Generic, non-grid search on hyper parameters.

    Randomized search is invoked with ``search_optimization`` set to
    ``"random"`` and behaves like scikit-learn's ``RandomizedSearchCV``.

    Bayesian search is invoked with ``search_optimization`` set to
    ``"bayesian"`` and behaves like scikit-learn's ``BayesSearchCV``.

    TuneSearchCV implements a "fit" and a "score" method.
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
            Serves as the ``param_distributions`` parameter in scikit-learn's
            ``RandomizedSearchCV`` or as the ``search_space`` parameter in
            ``BayesSearchCV``.

            For randomized search: dictionary with parameters names (string)
            as keys and distributions or lists of parameter settings to try
            for randomized search.

            Distributions must provide a rvs  method for sampling (such as
            those from scipy.stats.distributions).

            If a list is given, it is sampled uniformly. If a list of dicts is
            given, first a dict is sampled uniformly, and then a parameter is
            sampled using that dict as above.

            For Bayesian search: it is one of these cases:

            1. dictionary, where keys are parameter names (strings) and values
            are skopt.space.Dimension instances (Real, Integer or Categorical)
            or any other valid value that defines skopt dimension (see
            skopt.Optimizer docs). Represents search space over parameters of
            the provided estimator.

            2. list of dictionaries: a list of dictionaries, where every
            dictionary fits the description given in case 1 above. If a list of
            dictionary objects is given, then the search is performed
            sequentially for every parameter space with maximum number of
            evaluations set to self.n_iter.

            3. list of (dict, int > 0): an extension of case 2 above, where
            first element of every tuple is a dictionary representing some
            search subspace, similarly as in case 2, and second element is a
            number of iterations that will be spent optimizing over this
            subspace.

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

            Ignored when doing Bayesian search.

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

        search_optimization ("random" or "bayesian"):
            If "random", uses randomized search over the
            ``param_distributions``. If "bayesian", uses Bayesian
            optimization to search for hyperparameters.

    """

    def __init__(self,
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
                 early_stopping=False,
                 max_iters=10,
                 search_optimization="random"):

        if (search_optimization not in ["random", "bayesian"]
                and not isinstance(search_optimization, BayesOptSearch)):
            raise ValueError("Search optimization must be random or bayesian")
        if ((search_optimization == "bayesian"
             or isinstance(search_optimization, BayesOptSearch))
                and random_state is not None):
            warnings.warn(
                "random state is ignored when using Bayesian optimization")
        if isinstance(search_optimization, BayesOptSearch):
            search_optimization._metric = "average_test_score"
            warnings.warn("`param_distributions` is ignored when "
                          "passing in `BayesOptSearch` object")

        if isinstance(param_distributions, list):
            if search_optimization == "bayesian":
                raise ValueError("list of dictionaries for parameters "
                                 "is not supported for bayesian search")

        if isinstance(param_distributions, dict):
            check_param_distributions = [param_distributions]
        else:
            check_param_distributions = param_distributions
        for p in check_param_distributions:
            for dist in p.values():
                if search_optimization == "random":
                    if not (isinstance(dist, list) or hasattr(dist, "rvs")):
                        raise ValueError(
                            "distribution must be a list or scipy "
                            "distribution when using randomized search")
                else:
                    if not isinstance(dist, tuple):
                        raise ValueError(
                            "distribution must be a tuple when using "
                            "bayesian search")

        super(TuneSearchCV, self).__init__(
            estimator=estimator,
            scheduler=scheduler,
            scoring=scoring,
            n_jobs=n_jobs,
            cv=cv,
            verbose=verbose,
            refit=refit,
            error_score=error_score,
            return_train_score=return_train_score,
            early_stopping=early_stopping,
            max_iters=max_iters,
        )

        self.param_distributions = param_distributions
        self.num_samples = n_iter
        if search_optimization == "random":
            self.random_state = random_state
        self.search_optimization = search_optimization

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
        if (self.search_optimization == "bayesian"
                or isinstance(self.search_optimization, BayesOptSearch)):
            return

        if isinstance(self.param_distributions, list):
            return

        samples = 1
        all_lists = True
        for key, distribution in self.param_distributions.items():
            if isinstance(distribution, list):
                import random

                def get_sample(dist):
                    return lambda spec: dist[random.randint(0, len(dist) - 1)]

                config[key] = tune.sample_from(get_sample(distribution))
                samples *= len(distribution)
            else:
                all_lists = False

                def get_sample(dist):
                    return lambda spec: dist.rvs(1)[0]

                config[key] = tune.sample_from(get_sample(distribution))
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
        else:
            config["estimator"] = self.estimator

        if self.search_optimization == "random":
            if isinstance(self.param_distributions, list):
                analysis = tune.run(
                    _Trainable,
                    scheduler=self.scheduler,
                    search_alg=RandomListSearcher(self.param_distributions),
                    reuse_actors=True,
                    verbose=self.verbose,
                    stop={"training_iteration": self.max_iters},
                    num_samples=self.num_samples,
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
                    num_samples=self.num_samples,
                    config=config,
                    checkpoint_at_end=True,
                    resources_per_trial=resources_per_trial,
                )
        else:
            if self.search_optimization == "bayesian":
                search_algo = BayesOptSearch(
                    space=self.param_distributions,
                    metric="average_test_score",
                    utility_kwargs={
                        "kind": "ucb",
                        "kappa": 2.5,
                        "xi": 0.0
                    })
            else:
                search_algo = self.search_optimization

            analysis = tune.run(
                _Trainable,
                search_alg=search_algo,
                scheduler=self.scheduler,
                reuse_actors=True,
                verbose=self.verbose,
                stop={"training_iteration": self.max_iters},
                num_samples=self.num_samples,
                config=config,
                checkpoint_at_end=True,
                resources_per_trial=resources_per_trial,
            )

        return analysis
