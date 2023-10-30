<!-- markdownlint-disable -->

<a href="https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/tune_search.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `tune_search`
Class for cross-validation over distributions of hyperparameters 
-- Anthony Yu and Michael Chau 

**Global Variables**
---------------
- **available_optimizations**


---

<a href="https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/tune_search.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TuneSearchCV`
Generic, non-grid search on hyper parameters. 

Randomized search is invoked with ``search_optimization`` set to ``"random"`` and behaves like scikit-learn's ``RandomizedSearchCV``. 

Bayesian search can be invoked with several values of ``search_optimization``. 


 - ``"bayesian"``, using https://scikit-optimize.github.io/stable/ 
 - ``"bohb"``, using HpBandSter - https://github.com/automl/HpBandSter 

Tree-Parzen Estimators search is invoked with ``search_optimization`` set to ``"hyperopt"``, using HyperOpt - http://hyperopt.github.io/hyperopt 

All types of search aside from Randomized search require parent libraries to be installed. 

TuneSearchCV implements a "fit" and a "score" method. It also implements "predict", "predict_proba", "decision_function", "transform" and "inverse_transform" if they are implemented in the estimator used. 

The parameters of the estimator used to apply these methods are optimized by cross-validated search over parameter settings. 

In contrast to GridSearchCV, not all parameter values are tried out, but rather a fixed number of parameter settings is sampled from the specified distributions. The number of parameter settings that are tried is given by n_trials. 



**Args:**
 
 - <b>`estimator`</b> (`estimator`):  This is assumed to implement the  scikit-learn estimator interface. Either estimator needs to  provide a ``score`` function, or ``scoring`` must be passed. 
 - <b>`param_distributions`</b> (`dict` or `list` or `ConfigurationSpace`):  Serves  as the ``param_distributions`` parameter in scikit-learn's  ``RandomizedSearchCV`` or as the ``search_space`` parameter in  ``BayesSearchCV``. 
 - <b>`For randomized search`</b>:  dictionary with parameters names (string) as keys and distributions or lists of parameter settings to try for randomized search. Distributions must provide a rvs method for sampling (such as those from scipy.stats.distributions). Ray Tune search spaces are also supported. If a list is given, it is sampled uniformly. If a list of dicts is given, first a dict is sampled uniformly, and then a parameter is sampled using that dict as above. 
 - <b>`For other types of search`</b>:  dictionary with parameter names (string) as keys. Values can be 


        - a (lower_bound, upper_bound) tuple (for Real or Integer params) 
        - a (lower_bound, upper_bound, "prior") tuple (for Real params) 
        - as a list of categories (for Categorical dimensions) 
        - Ray Tune search space (eg. ``tune.uniform``) 

``"bayesian"`` (scikit-optimize) also accepts 


        - skopt.space.Dimension instance (Real, Integer or Categorical). 

``"hyperopt"`` (HyperOpt) also accepts 


        - an instance of a hyperopt.pyll.base.Apply object. 

``"bohb"`` (HpBandSter) also accepts 


        - ConfigSpace.hyperparameters.Hyperparameter instance. 

``"optuna"`` (Optuna) also accepts 


        - an instance of a optuna.distributions.BaseDistribution object. 

For ``"bohb"`` (HpBandSter) it is also possible to pass a `ConfigSpace.ConfigurationSpace` object instead of dict or a list. 


 - <b>`https`</b>: //scikit-optimize.github.io/stable/modules/ classes.html#module-skopt.space.space 
 - <b>`early_stopping`</b> (bool, str or :class:`TrialScheduler`, optional):  Option  to stop fitting to a hyperparameter configuration if it performs  poorly. Possible inputs are: 


        - If True, defaults to ASHAScheduler. 
        - A string corresponding to the name of a Tune Trial Scheduler  (i.e., "ASHAScheduler"). To specify parameters of the scheduler,  pass in a scheduler object instead of a string. 
        - Scheduler for executing fit with early stopping. Only a subset  of schedulers are currently supported. The scheduler will only be  used if the estimator supports partial fitting 
        - If None or False, early stopping will not be used. 

 Unless a ``HyperBandForBOHB`` object is passed,  this parameter is ignored for ``"bohb"``, as it requires  ``HyperBandForBOHB``. 


 - <b>`n_trials`</b> (int):  Number of parameter settings that are sampled.  n_trials trades off runtime vs quality of the solution.  Defaults to 10. 
 - <b>`scoring`</b> (str, callable, list/tuple, dict, or None):  A single  string or a callable to evaluate the predictions on the test set. 
 - <b>`See https`</b>: //scikit-learn.org/stable/modules/model_evaluation.html #scoring-parameter for all options. For evaluating multiple metrics, either give a list/tuple of (unique) strings or a dict with names as keys and callables as values. If None, the estimator's score method is used. Defaults to None. 
 - <b>`n_jobs`</b> (int):  Number of jobs to run in parallel. None or -1 means  using all processors. Defaults to None. If set to 1, jobs  will be run using Ray's 'local mode'. This can  lead to significant speedups if the model takes < 10 seconds  to fit due to removing inter-process communication overheads. 
 - <b>`refit`</b> (bool, str, or `callable`):  Refit an estimator using the  best found parameters on the whole dataset.  For multiple metric evaluation, this needs to be a string denoting  the scorer that would be used to find the best parameters for  refitting the estimator at the end.  The refitted estimator is made available at the ``best_estimator_``  attribute and permits using ``predict`` directly on this  ``GridSearchCV`` instance.  Also for multiple metric evaluation, the attributes  ``best_index_``, ``best_score_`` and ``best_params_`` will only be  available if ``refit`` is set and all of them will be determined  w.r.t this specific scorer. If refit not needed, set to False.  See ``scoring`` parameter to know more about multiple metric  evaluation. Defaults to True. 
 - <b>`cv`</b> (int, `cross-validation generator` or `iterable`):  Determines  the cross-validation splitting strategy.  Possible inputs for cv are: 


        - None, to use the default 5-fold cross validation, 
        - integer, to specify the number of folds in a `(Stratified)KFold`, 
        - An iterable yielding (train, test) splits as arrays of indices. 

 For integer/None inputs, if the estimator is a classifier and ``y`` 
 - <b>`is either binary or multiclass, `</b>: class:`StratifiedKFold` is used. 
 - <b>`In all other cases, `</b>: class:`KFold` is used. Defaults to None. 
 - <b>`verbose`</b> (int):  Controls the verbosity: 0 = silent, 1 = only status  updates, 2 = status and trial results. Defaults to 0. 
 - <b>`random_state`</b> (int or `RandomState`):  Pseudo random number generator  state used for random uniform  sampling from lists of possible values instead of scipy.stats  distributions.  If int, random_state is the seed used by the random number  generator;  If RandomState instance, a seed is sampled from random_state;  If None, the random number generator is the RandomState instance  used by np.random and no seed is provided. Defaults to None.  Ignored when using BOHB. 
 - <b>`error_score`</b> ('raise' or int or float):  Value to assign to the score if  an error occurs in estimator  fitting. If set to 'raise', the error is raised. If a numeric value  is given, FitFailedWarning is raised. This parameter does not  affect the refit step, which will always raise the error.  Defaults to np.nan. 
 - <b>`return_train_score`</b> (bool):  If ``False``, the ``cv_results_``  attribute will not include training scores. Defaults to False.  Computing training scores is used to get insights on how different  parameter settings impact the overfitting/underfitting trade-off.  However computing the scores on the training set can be  computationally expensive and is not strictly required to select  the parameters that yield the best generalization performance. 
 - <b>`local_dir`</b> (str):  A string that defines where checkpoints and logs will  be stored. Defaults to "~/ray_results" name (str) – Name of experiment (for Ray Tune). 
 - <b>`max_iters`</b> (int):  Indicates the maximum number of epochs to run for each  hyperparameter configuration sampled (specified by ``n_trials``).  This parameter is used for early stopping. Defaults to 1.  Depending on the classifier type provided, a resource parameter  (`resource_param = max_iter or n_estimators`) will be detected.  The value of `resource_param` will be treated as a  "max resource value", and all classifiers will be  initialized with `max resource value // max_iters`, where  `max_iters` is this defined parameter. On each epoch,  resource_param (max_iter or n_estimators) is  incremented by `max resource value // max_iters`. search_optimization ("random" or "bayesian" or "bohb" or "hyperopt"  or "optuna" or `ray.tune.search.Searcher` instance):  Randomized search is invoked with ``search_optimization`` set to  ``"random"`` and behaves like scikit-learn's  ``RandomizedSearchCV``. 

 Bayesian search can be invoked with several values of  ``search_optimization``. 


        - ``"bayesian"`` via https://scikit-optimize.github.io/stable/ 
        - ``"bohb"`` via http://github.com/automl/HpBandSter 

 Tree-Parzen Estimators search is invoked with  ``search_optimization`` set to ``"hyperopt"`` via HyperOpt: 
 - <b>`http`</b>: //hyperopt.github.io/hyperopt 

All types of search aside from Randomized search require parent libraries to be installed. 

Alternatively, instead of a string, a Ray Tune Searcher instance can be used, which will be passed to ``tune.run()``. 
 - <b>`use_gpu`</b> (bool):  Indicates whether to use gpu for fitting.  Defaults to False. If True, training will start processes  with the proper CUDA VISIBLE DEVICE settings set. If a Ray  cluster has been initialized, all available GPUs will  be used. 
 - <b>`loggers`</b> (list):  A list of the names of the Tune loggers as strings  to be used to log results. Possible values are "tensorboard",  "csv", "mlflow", and "json" 
 - <b>`pipeline_auto_early_stop`</b> (bool):  Only relevant if estimator is Pipeline  object and early_stopping is enabled/True. If True, early stopping  will be performed on the last stage of the pipeline (which must  support early stopping). If False, early stopping will be  determined by 'Pipeline.warm_start' or 'Pipeline.partial_fit'  capabilities, which are by default not supported by standard  SKlearn. Defaults to True. 
 - <b>`stopper`</b> (ray.tune.stopper.Stopper):  Stopper objects passed to  ``tune.run()``. 
 - <b>`time_budget_s`</b> (int|float|datetime.timedelta):  Global time budget in  seconds after which all trials are stopped. Can also be a  ``datetime.timedelta`` object. The stopping condition is checked  after receiving a result, i.e. after each training iteration. 
 - <b>`mode`</b> (str):  One of {min, max}. Determines whether objective is  minimizing or maximizing the metric attribute. Defaults to "max". search_kwargs (dict):  Additional arguments to pass to the SearchAlgorithms (tune.suggest)  objects. 

<a href="https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/tune_search.py#L306"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    estimator,
    param_distributions,
    early_stopping=None,
    n_trials=10,
    scoring=None,
    n_jobs=None,
    refit=True,
    cv=None,
    verbose=0,
    random_state=None,
    error_score=nan,
    return_train_score=False,
    local_dir=None,
    name=None,
    max_iters=1,
    search_optimization='random',
    use_gpu=False,
    loggers=None,
    pipeline_auto_early_stop=True,
    stopper=None,
    time_budget_s=None,
    sk_n_jobs=None,
    mode=None,
    search_kwargs=None,
    **kwargs
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
