def has_xgboost():
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


def is_xgboost_model(clf):
    if not has_xgboost():
        return False
    from xgboost.sklearn import XGBModel  # noqa: F401
    return isinstance(clf, XGBModel)


def has_lightgbm():
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False


def is_lightgbm_model(clf):
    if not has_xgboost():
        return False
    from lightgbm.sklearn import LGBMModel  # noqa: F401
    return isinstance(clf, LGBMModel)


def has_catboost():
    try:
        import catboost  # noqa: F401
        return True
    except ImportError:
        return False


def is_catboost_model(clf):
    if not has_xgboost():
        return False
    from catboost import CatBoost  # noqa: F401
    return isinstance(clf, CatBoost)
