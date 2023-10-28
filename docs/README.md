<!-- markdownlint-disable -->

# API Overview

## Modules

- [`list_searcher`](./list_searcher.md#module-list_searcher): Helper class to support passing a
- [`tune_basesearch`](./tune_basesearch.md#module-tune_basesearch): Parent class for a cross-validation interface
- [`tune_gridsearch`](./tune_gridsearch.md#module-tune_gridsearch): Class for doing grid search over lists of hyperparameters
- [`tune_search`](./tune_search.md#module-tune_search): Class for cross-validation over distributions of hyperparameters
- [`utils`](./utils.md#module-utils)

## Classes

- [`list_searcher.ListSearcher`](./list_searcher.md#class-listsearcher): Custom search algorithm to support passing in a list of
- [`list_searcher.RandomListSearcher`](./list_searcher.md#class-randomlistsearcher): Custom search algorithm to support passing in a list of
- [`tune_basesearch.TuneBaseSearchCV`](./tune_basesearch.md#class-tunebasesearchcv): Abstract base class for TuneGridSearchCV and TuneSearchCV
- [`tune_gridsearch.TuneGridSearchCV`](./tune_gridsearch.md#class-tunegridsearchcv): Exhaustive search over specified parameter values for an estimator.
- [`tune_search.TuneSearchCV`](./tune_search.md#class-tunesearchcv): Generic, non-grid search on hyper parameters.
- [`utils.EarlyStopping`](./utils.md#class-earlystopping): An enumeration.

## Functions

- [`tune_basesearch.resolve_early_stopping`](./tune_basesearch.md#function-resolve_early_stopping)
- [`utils.check_error_warm_start`](./utils.md#function-check_error_warm_start)
- [`utils.check_is_pipeline`](./utils.md#function-check_is_pipeline)
- [`utils.check_partial_fit`](./utils.md#function-check_partial_fit)
- [`utils.check_warm_start_ensemble`](./utils.md#function-check_warm_start_ensemble)
- [`utils.check_warm_start_iter`](./utils.md#function-check_warm_start_iter)
- [`utils.get_early_stop_type`](./utils.md#function-get_early_stop_type)
- [`utils.is_tune_grid_search`](./utils.md#function-is_tune_grid_search): Checks if obj is a dictionary returned by tune.grid_search.
- [`utils.resolve_logger_callbacks`](./utils.md#function-resolve_logger_callbacks)


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
