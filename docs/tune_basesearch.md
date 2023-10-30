<!-- markdownlint-disable -->

<a href="https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/tune_basesearch.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `tune_basesearch`
Parent class for a cross-validation interface built with a Ray Tune back-end. 

Implementation derived from referencing the equivalent GridSearchCV interfaces from Dask and Optuna. 

https://ray.readthedocs.io/en/latest/tune.html https://dask.org https://optuna.org 
    -- Anthony Yu and Michael Chau 

**Global Variables**
---------------
- **DEFAULT_MODE**

---

<a href="https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/tune_basesearch.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `resolve_early_stopping`

```python
resolve_early_stopping(early_stopping, max_iters, metric_name)
```






---

<a href="https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/tune_basesearch.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TuneBaseSearchCV`
Abstract base class for TuneGridSearchCV and TuneSearchCV 

<a href="https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/tune_basesearch.py#L320"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    estimator,
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

<a href="https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/tune_basesearch.py#L602"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y=None, groups=None, tune_params=None, **fit_params)
```

Run fit with all sets of parameters. 

``tune.run`` is used to perform the fit procedure. 



**Args:**
 
 - <b>`X (`</b>: obj:`array-like` (shape = [n_samples, n_features])):  Training vector, where n_samples is the number of samples and  n_features is the number of features. 
 - <b>`y`</b> (:obj:`array-like`):  Shape of array expected to be [n_samples]  or [n_samples, n_output]). Target relative to X for  classification or regression; None for unsupervised learning. 
 - <b>`groups (`</b>: obj:`array-like` (shape (n_samples,)), optional):  Group labels for the samples used while splitting the dataset  into train/test set. Only used in conjunction with a "Group"  `cv` instance (e.g., `GroupKFold`). 
 - <b>`tune_params (`</b>: obj:`dict`, optional):  Parameters passed to ``tune.run`` used for parameter search. 
 - <b>`**fit_params (`</b>: obj:`dict` of str): Parameters passed to  the ``fit`` method of the estimator. 



**Returns:**
 
 - <b>`:obj`</b>: `TuneBaseSearchCV` child instance, after fitting. 

---

<a href="https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/tune_basesearch.py#L629"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `score`

```python
score(X, y=None)
```

Compute the score(s) of an estimator on a given test set. 



**Args:**
 
 - <b>`X`</b> (:obj:`array-like` (shape = [n_samples, n_features])):  Input  data, where n_samples is the number of samples and  n_features is the number of features. 
 - <b>`y`</b> (:obj:`array-like`):  Shape of array is expected to be  [n_samples] or [n_samples, n_output]). Target relative to X  for classification or regression. You can also pass in  None for unsupervised learning. 



**Returns:**
 
 - <b>`float`</b>:  computed score 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
