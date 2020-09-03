def check_partial_fit(estimator):
    return callable(getattr(estimator, "partial_fit", None))


def check_warm_start(estimator):
    from sklearn.tree import BaseDecisionTree
    from sklearn.ensemble import BaseEnsemble
    is_not_tree_subclass = not issubclass(type(estimator), BaseDecisionTree)
    is_not_ensemble_subclass = not issubclass(type(estimator), BaseEnsemble)

    return (hasattr(estimator, "warm_start")
            and hasattr(estimator, "max_iter") and is_not_ensemble_subclass
            and is_not_tree_subclass)
