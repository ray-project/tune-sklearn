def has_xgboost():
    try:
        import xgboost  # ignore: F401
        return True
    except ImportError:
        return False

def is_xgboost_model(clf):
    if not has_xgboost():
        return False
    import xgboost  # ignore: F401
    return isinstance(clf, XGBModel)
