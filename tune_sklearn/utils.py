from sklearn.metrics import check_scoring
import numpy as np


def check_partial_fit(estimator):
    return callable(getattr(estimator, "partial_fit", None))


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
