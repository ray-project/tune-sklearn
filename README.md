# tune-sklearn
Tune-sklearn is a package that integrates Ray Tune's hyperparameter tuning and scikit-learn's models, allowing users to optimize hyerparameter searching for sklearn using Tune's schedulers. Tune-sklearn follows the same API as scikit-learn's GridSearchCV, but allows for more flexibility in defining hyperparameter search regions, such as distributions to sample from.

## Quick Start
Use tune-sklearn TuneRandomizedSearchCV to tune sklearn model

```python
iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

clf = SGDClassifier()
param_grid = {
    'n_estimators': scipy.stats.randint(20, 80),
    'alpha': scipy.stats.uniform(1e-4, 1e-1)
}

scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="average_test_score",
            mode="max",
            perturbation_interval=5,
            resample_probability=1.0,
            hyperparam_mutations = {
                "alpha" : lambda: np.random.choice([1e-4, 1e-3, 1e-2, 1e-1])
            })

tune_search = TuneRandomizedSearchCV(clf, 
            param_grid=param_grid,
            scheduler=scheduler,
            n_jobs=5,
            refit=True,
            early_stopping=True,
            iters=10)
tune_search.fit(x_train, y_train)
```

Use tune-sklearn TuneGridSearchCV to tune sklearn model
```python
# Loading the Digits dataset
digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = {'kernel': ['rbf'], 
                    'gamma': [1e-3, 1e-4],
                    'C': [1, 10, 100, 1000]
                    }

tune_search = TuneGridSearchCV(SVC(),  
                               tuned_parameters, 
                               scheduler=MedianStoppingRule(), 
                               iters=20)
tune_search.fit(X_train, y_train)

pred = tune_search.predict(X_test)
```
## More information
[Ray Tune](https://ray.readthedocs.io/en/latest/tune.html)
