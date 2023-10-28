<!-- markdownlint-disable -->

<a href="https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/list_searcher.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `list_searcher`
Helper class to support passing a list of dictionaries for hyperparameters 
    -- Anthony Yu and Michael Chau 



---

<a href="https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/list_searcher.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ListSearcher`
Custom search algorithm to support passing in a list of dictionaries to TuneGridSearchCV 

<a href="https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/list_searcher.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(param_grid)
```






---

#### <kbd>property</kbd> metric

The training result objective value attribute. 

---

#### <kbd>property</kbd> mode

Specifies if minimizing or maximizing the metric. 



---

<a href="https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/list_searcher.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `on_trial_complete`

```python
on_trial_complete(**kwargs)
```





---

<a href="https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/list_searcher.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `suggest`

```python
suggest(trial_id)
```






---

<a href="https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/list_searcher.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RandomListSearcher`
Custom search algorithm to support passing in a list of dictionaries to TuneSearchCV for randomized search 

<a href="https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/list_searcher.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(param_grid)
```






---

#### <kbd>property</kbd> metric

The training result objective value attribute. 

---

#### <kbd>property</kbd> mode

Specifies if minimizing or maximizing the metric. 



---

<a href="https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/list_searcher.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `on_trial_complete`

```python
on_trial_complete(**kwargs)
```





---

<a href="https://github.com/ray-project/tune-sklearn/blob/master/tune_sklearn/list_searcher.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `suggest`

```python
suggest(trial_id)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
