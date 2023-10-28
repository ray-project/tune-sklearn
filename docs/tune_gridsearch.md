<!-- markdownlint-disable -->

<a href="https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/tune_gridsearch.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `tune_gridsearch`
Class for doing grid search over lists of hyperparameters 
-- Anthony Yu and Michael Chau 



---

<a href="https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/tune_gridsearch.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TuneGridSearchCV`
Exhaustive search over specified parameter values for an estimator. 

Important members are fit, predict. 

GridSearchCV implements a "fit" and a "score" method. It also implements "predict", "predict_proba", "decision_function", "transform" and "inverse_transform" if they are implemented in the estimator used. 

The parameters of the estimator used to apply these methods are optimized by cross-validated grid-search over a parameter grid. 



**Args:**
 
 - <b>`estimator`</b> (`estimator`):  Object that implements the  scikit-learn estimator interface. Either estimator needs to  provide a ``score`` function, or ``scoring`` must be passed. 
 - <b>`param_grid`</b> (`dict` or `list` of `dict`):  Dictionary with parameters  names (string) as keys and lists or tune.grid_search outputs of  parameter settings to try as values, or a list of such  dictionaries, in which case the grids spanned by each dictionary  in the list are explored. This enables searching over any sequence  of parameter settings. 
 - <b>`early_stopping`</b> (bool, str or :class:`TrialScheduler`, optional):  Option  to stop fitting to a hyperparameter configuration if it performs  poorly. Possible inputs are: 


        - If True, defaults to ASHAScheduler. 
        - A string corresponding to the name of a Tune Trial Scheduler  (i.e., "ASHAScheduler"). To specify parameters of the scheduler,  pass in a scheduler object instead of a string. 
        - Scheduler for executing fit with early stopping. Only a subset  of schedulers are currently supported. The scheduler will only be  used if the estimator supports partial fitting 
        - If None or False, early stopping will not be used. 


 - <b>`scoring`</b> (str, list/tuple, dict, or None):  A single  string or a callable to evaluate the predictions on the test set. 
 - <b>`See https`</b>: //scikit-learn.org/stable/modules/model_evaluation.html #scoring-parameter for all options. For evaluating multiple metrics, either give a list/tuple of (unique) strings or a dict with names as keys and callables as values. If None, the estimator's score method is used. Defaults to None. 
 - <b>`n_jobs`</b> (int):  Number of jobs to run in parallel. None or -1 means  using all processors. Defaults to None. If set to 1, jobs  will be run using Ray's 'local mode'. This can  lead to significant speedups if the model takes < 10 seconds  to fit due to removing inter-process communication overheads. 
 - <b>`cv`</b> (int, `cross-validation generator` or `iterable`):  Determines the  cross-validation splitting strategy. Possible inputs for cv are: 


        - None, to use the default 5-fold cross validation, 
        - integer, to specify the number of folds in a `(Stratified)KFold`, 
        - An iterable yielding (train, test) splits as arrays of indices. 

 For integer/None inputs, if the estimator is a classifier and ``y`` 
 - <b>`is either binary or multiclass, `</b>: class:`StratifiedKFold` is used. 
 - <b>`In all other cases, `</b>: class:`KFold` is used. Defaults to None. refit (bool or str): Refit an estimator using the best found parameters on the whole dataset. For multiple metric evaluation, this needs to be a string denoting the scorer that would be used to find the best parameters for refitting the estimator at the end. The refitted estimator is made available at the ``best_estimator_`` attribute and permits using ``predict`` directly on this ``GridSearchCV`` instance. Also for multiple metric evaluation, the attributes ``best_index_``, ``best_score_`` and ``best_params_`` will only be available if ``refit`` is set and all of them will be determined w.r.t this specific scorer. If refit not needed, set to False. See ``scoring`` parameter to know more about multiple metric evaluation. Defaults to True. 
 - <b>`verbose`</b> (int):  Controls the verbosity: 0 = silent, 1 = only  status updates, 2 = status and trial results. Defaults to 0. 
 - <b>`error_score`</b> ('raise' or int or float):  Value to assign to the score  if an error occurs in estimator fitting. If set to 'raise',  the error is raised. If a numeric value  is given, FitFailedWarning is raised. This parameter does not  affect the refit step, which will always raise the error.  Defaults to np.nan. 
 - <b>`return_train_score`</b> (bool):  If ``False``, the ``cv_results_``  attribute will not include training scores. Defaults to False.  Computing training scores is used to get insights on how different  parameter settings impact the overfitting/underfitting trade-off.  However computing the scores on the training set can be  computationally expensive and is not strictly required to select  the parameters that yield the best generalization performance. 
 - <b>`local_dir`</b> (str):  A string that defines where checkpoints will  be stored. Defaults to "~/ray_results". name (str) – Name of experiment (for Ray Tune). 
 - <b>`max_iters`</b> (int):  Indicates the maximum number of epochs to run for each  hyperparameter configuration sampled.  This parameter is used for early stopping. Defaults to 1.  Depending on the classifier type provided, a resource parameter  (`resource_param = max_iter or n_estimators`) will be detected.  The value of `resource_param` will be treated as a  "max resource value", and all classifiers will be  initialized with `max resource value // max_iters`, where  `max_iters` is this defined parameter. On each epoch,  resource_param (max_iter or n_estimators) is  incremented by `max resource value // max_iters`. 
 - <b>`use_gpu`</b> (bool):  Indicates whether to use gpu for fitting.  Defaults to False. If True, training will start processes  with the proper CUDA VISIBLE DEVICE settings set. If a Ray  cluster has been initialized, all available GPUs will  be used. 
 - <b>`loggers`</b> (list):  A list of the names of the Tune loggers as strings  to be used to log results. Possible values are "tensorboard",  "csv", "mlflow", and "json" 
 - <b>`pipeline_auto_early_stop`</b> (bool):  Only relevant if estimator is Pipeline  object and early_stopping is enabled/True. If True, early stopping  will be performed on the last stage of the pipeline (which must  support early stopping). If False, early stopping will be  determined by 'Pipeline.warm_start' or 'Pipeline.partial_fit'  capabilities, which are by default not supported by standard  SKlearn. Defaults to True. 
 - <b>`stopper`</b> (ray.tune.stopper.Stopper):  Stopper objects passed to  ``tune.run()``. 
 - <b>`time_budget_s`</b> (int|float|datetime.timedelta):  Global time budget in  seconds after which all trials are stopped. Can also be a  ``datetime.timedelta`` object. 
 - <b>`mode`</b> (str):  One of {min, max}. Determines whether objective is  minimizing or maximizing the metric attribute. Defaults to "max". 

<a href="https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/tune_gridsearch.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    estimator,
    param_grid,
    early_stopping=None,
    scoring=None,
    n_jobs=None,
    cv=5,
    refit=True,
    verbose=0,
    error_score='raise',
    return_train_score=False,
    local_dir=None,
    name=None,
    max_iters=1,
    use_gpu=False,
    loggers=None,
    pipeline_auto_early_stop=True,
    stopper=None,
    time_budget_s=None,
    sk_n_jobs=None,
    mode=None
)
```






---

#### <kbd>property</kbd> best_estimator_

estimator: Estimator that was chosen by the search, i.e. estimator which gave highest score (or smallest loss if specified) on the left out data. Not available if ``refit=False``. 

See ``refit`` parameter for more information on allowed values. 

---

#### <kbd>property</kbd> best_index_

int: The index (of the ``cv_results_`` arrays) which corresponds to the best candidate parameter setting. 

The dict at ``search.cv_results_['params'][search.best_index_]`` gives the parameter setting for the best model, that gives the highest mean score (``search.best_score_``). 

For multi-metric evaluation, this is present only if ``refit`` is specified. 

---

#### <kbd>property</kbd> best_params_

dict: Parameter setting that gave the best results on the hold out data. 

For multi-metric evaluation, this is present only if ``refit`` is specified. 

---

#### <kbd>property</kbd> best_score_

float: Mean cross-validated score of the best_estimator 

For multi-metric evaluation, this is present only if ``refit`` is specified. 

---

#### <kbd>property</kbd> classes_

list: Get the list of unique classes found in the target `y`. 

---

#### <kbd>property</kbd> decision_function

function: Get decision_function on the estimator with the best found parameters. 

Only available if ``refit=True`` and the underlying estimator supports ``decision_function``. 

---

#### <kbd>property</kbd> inverse_transform

function: Get inverse_transform on the estimator with the best found parameters. 

Only available if the underlying estimator implements ``inverse_transform`` and ``refit=True``. 

---

#### <kbd>property</kbd> multimetric_

bool: Whether evaluation performed was multi-metric. 

---

#### <kbd>property</kbd> n_features_in_

Number of features seen during :term:`fit`. 

Only available when `refit=True`. 

---

#### <kbd>property</kbd> n_splits_

int: The number of cross-validation splits (folds/iterations). 

---

#### <kbd>property</kbd> predict

function: Get predict on the estimator with the best found parameters. 

Only available if ``refit=True`` and the underlying estimator supports ``predict``. 

---

#### <kbd>property</kbd> predict_log_proba

function: Get predict_log_proba on the estimator with the best found parameters. 

Only available if ``refit=True`` and the underlying estimator supports ``predict_log_proba``. 

---

#### <kbd>property</kbd> predict_proba

function: Get predict_proba on the estimator with the best found parameters. 

Only available if ``refit=True`` and the underlying estimator supports ``predict_proba``. 

---

#### <kbd>property</kbd> refit_time_

float: Seconds used for refitting the best model on the whole dataset. 

This is present only if ``refit`` is not False. 

---

#### <kbd>property</kbd> scorer_

function or a dict: Scorer function used on the held out data to choose the best parameters for the model. 

For multi-metric evaluation, this attribute holds the validated ``scoring`` dict which maps the scorer key to the scorer callable. 

---

#### <kbd>property</kbd> transform

function: Get transform on the estimator with the best found parameters. 

Only available if the underlying estimator supports ``transform`` and ``refit=True``. 






---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
