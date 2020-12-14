from collections import defaultdict
from typing import Dict

from sklearn.metrics import check_scoring
from sklearn.pipeline import Pipeline
from tune_sklearn._detect_booster import (
    is_xgboost_model, is_lightgbm_model_of_required_version, is_catboost_model)
import numpy as np
from enum import Enum, auto
from collections.abc import Sequence

try:
    from ray.tune.stopper import MaximumIterationStopper
except ImportError:
    from ray.tune.stopper import Stopper

    class MaximumIterationStopper(Stopper):
        def __init__(self, max_iter: int):
            self._max_iter = max_iter
            self._iter = defaultdict(lambda: 0)

        def __call__(self, trial_id: str, result: Dict):
            self._iter[trial_id] += 1
            return self._iter[trial_id] >= self._max_iter

        def stop_all(self):
            return False


class EarlyStopping(Enum):
    PARTIAL_FIT = auto()
    WARM_START_ITER = auto()
    WARM_START_ENSEMBLE = auto()
    XGB = auto()
    LGBM = auto()
    CATBOOST = auto()
    NO_EARLY_STOP = auto()


def check_partial_fit(estimator):
    return callable(getattr(estimator, "partial_fit", None))


def check_is_pipeline(estimator):
    return isinstance(estimator, Pipeline)


def check_warm_start_iter(estimator):
    from sklearn.tree import BaseDecisionTree
    from sklearn.ensemble import BaseEnsemble
    is_not_tree_subclass = not issubclass(type(estimator), BaseDecisionTree)
    is_not_ensemble_subclass = not issubclass(type(estimator), BaseEnsemble)

    return (hasattr(estimator, "warm_start")
            and hasattr(estimator, "max_iter") and is_not_ensemble_subclass
            and is_not_tree_subclass)


def check_warm_start_ensemble(estimator):
    from sklearn.ensemble import BaseEnsemble
    is_ensemble_subclass = issubclass(type(estimator), BaseEnsemble)

    return (hasattr(estimator, "warm_start")
            and hasattr(estimator, "n_estimators") and is_ensemble_subclass)


def check_error_warm_start(early_stop_type, estimator_config, estimator):
    if check_is_pipeline(estimator):
        if (early_stop_type == EarlyStopping.WARM_START_ITER
                and f"{estimator.steps[-1][0]}__max_iter" in estimator_config):
            raise ValueError("tune-sklearn uses `max_iter` to warm "
                             "start, so this parameter can't be "
                             "set when warm start early stopping. ")

        if (early_stop_type == EarlyStopping.WARM_START_ENSEMBLE and
                f"{estimator.steps[-1][0]}__n_estimators" in estimator_config):
            raise ValueError("tune-sklearn uses `n_estimators` to warm "
                             "start, so this parameter can't be "
                             "set when warm start early stopping. ")
    else:
        if (early_stop_type == EarlyStopping.WARM_START_ITER
                and "max_iter" in estimator_config):
            raise ValueError("tune-sklearn uses `max_iter` to warm "
                             "start, so this parameter can't be "
                             "set when warm start early stopping. ")
        if (early_stop_type == EarlyStopping.WARM_START_ENSEMBLE
                and "n_estimators" in estimator_config):
            raise ValueError("tune-sklearn uses `n_estimators` to warm "
                             "start, so this parameter can't be "
                             "set when warm start early stopping. ")


def get_early_stop_type(estimator, early_stopping):
    if not early_stopping:
        return EarlyStopping.NO_EARLY_STOP
    can_partial_fit = check_partial_fit(estimator)
    can_warm_start_iter = check_warm_start_iter(estimator)
    can_warm_start_ensemble = check_warm_start_ensemble(estimator)
    is_xgb = is_xgboost_model(estimator)
    is_lgbm = is_lightgbm_model_of_required_version(estimator)
    is_catboost = is_catboost_model(estimator)
    if is_xgb:
        return EarlyStopping.XGB
    elif is_lgbm:
        return EarlyStopping.LGBM
    elif is_catboost:
        return EarlyStopping.CATBOOST
    elif can_partial_fit:
        return EarlyStopping.PARTIAL_FIT
    elif can_warm_start_iter:
        return EarlyStopping.WARM_START_ITER
    elif can_warm_start_ensemble:
        return EarlyStopping.WARM_START_ENSEMBLE
    else:
        return EarlyStopping.NO_EARLY_STOP


def _aggregate_score_dicts(scores):
    """Aggregate the list of dict to dict of np ndarray
    The aggregated output of _fit_and_score will be a list of dict
    of form [{'prec': 0.1, 'acc':1.0}, {'prec': 0.1, 'acc':1.0}, ...]
    Convert it to a dict of array {'prec': np.array([0.1 ...]), ...}
    Parameters
    ----------
    scores : list of dict
        List of dicts of the scores for all scorers. This is a flat list,
        assumed originally to be of row major order.
    Example
    -------
    >>> scores = [{'a': 1, 'b':10}, {'a': 2, 'b':2}, {'a': 3, 'b':3},
    ...           {'a': 10, 'b': 10}]                         # doctest: +SKIP
    >>> _aggregate_score_dicts(scores)                        # doctest: +SKIP
    {'a': array([1, 2, 3, 10]),
     'b': array([10, 2, 3, 10])}
    """
    return {
        key: np.asarray([score[key] for score in scores])
        for key in scores[0]
    }


def _check_multimetric_scoring(estimator, scoring=None):
    """Check the scoring parameter in cases when multiple metrics are allowed
    Parameters
    ----------
    estimator : sklearn estimator instance
        The estimator for which the scoring will be applied.
    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.
        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.
        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.
        See :ref:`multimetric_grid_search` for an example.
        If None the estimator's score method is used.
        The return value in that case will be ``{'score': <default_scorer>}``.
        If the estimator's score method is not available, a ``TypeError``
        is raised.
    Returns
    -------
    scorers_dict : dict
        A dict mapping each scorer name to its validated scorer.
    is_multimetric : bool
        True if scorer is a list/tuple or dict of callables
        False if scorer is None/str/callable
    """
    if callable(scoring) or scoring is None or isinstance(scoring, str):
        scorers = {"score": check_scoring(estimator, scoring=scoring)}
        return scorers, False
    else:
        err_msg_generic = (
            "scoring should either be a single string or "
            "callable for single metric evaluation or a "
            "list/tuple of strings or a dict of scorer name "
            "mapped to the callable for multiple metric "
            "evaluation. Got %s of type %s" % (repr(scoring), type(scoring)))

        if isinstance(scoring, (list, tuple, set)):
            err_msg = ("The list/tuple elements must be unique "
                       "strings of predefined scorers. ")
            invalid = False
            try:
                keys = set(scoring)
            except TypeError:
                invalid = True
            if invalid:
                raise ValueError(err_msg)

            if len(keys) != len(scoring):
                raise ValueError(err_msg + "Duplicate elements were found in"
                                 " the given list. %r" % repr(scoring))
            elif len(keys) > 0:
                if not all(isinstance(k, str) for k in keys):
                    if any(callable(k) for k in keys):
                        raise ValueError(err_msg +
                                         "One or more of the elements were "
                                         "callables. Use a dict of score name "
                                         "mapped to the scorer callable. "
                                         "Got %r" % repr(scoring))
                    else:
                        raise ValueError(
                            err_msg + "Non-string types were found in "
                            "the given list. Got %r" % repr(scoring))
                scorers = {
                    scorer: check_scoring(estimator, scoring=scorer)
                    for scorer in scoring
                }
            else:
                raise ValueError(err_msg +
                                 "Empty list was given. %r" % repr(scoring))

        elif isinstance(scoring, dict):
            keys = set(scoring)
            if not all(isinstance(k, str) for k in keys):
                raise ValueError("Non-string types were found in the keys of "
                                 "the given dict. scoring=%r" % repr(scoring))
            if len(keys) == 0:
                raise ValueError(
                    "An empty dict was passed. %r" % repr(scoring))
            scorers = {
                key: check_scoring(estimator, scoring=scorer)
                for key, scorer in scoring.items()
            }
        else:
            raise ValueError(err_msg_generic)
        return scorers, True


def is_tune_grid_search(obj):
    """Checks if obj is a dictionary returned by tune.grid_search.
    Returns bool.
    """
    return isinstance(
        obj, dict) and len(obj) == 1 and "grid_search" in obj and isinstance(
            obj["grid_search"], list)


# adapted from sklearn.model_selection._search
def _check_param_grid_tune_grid_search(param_grid):
    if hasattr(param_grid, "items"):
        param_grid = [param_grid]

    for p in param_grid:
        for name, v in p.items():
            if is_tune_grid_search(v):
                continue

            if isinstance(v, np.ndarray) and v.ndim > 1:
                raise ValueError("Parameter array should be one-dimensional.")

            if (isinstance(v, str) or not isinstance(v,
                                                     (np.ndarray, Sequence))):
                raise ValueError("Parameter grid for parameter ({0}) needs to"
                                 " be a tune.grid_search, list or numpy array,"
                                 " but got ({1})."
                                 " Single values need to be wrapped in a list"
                                 " with one element.".format(name, type(v)))

            if len(v) == 0:
                raise ValueError("Parameter values for parameter ({0}) need "
                                 "to be a non-empty sequence.".format(name))
