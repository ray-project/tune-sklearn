from tune_sklearn.tune_search import TuneRandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.stats import uniform
from ray.tune.schedulers import MedianStoppingRule
import numpy as np

iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

clf = SGDClassifier()
param_distributions = {
    'alpha': uniform(1e-4, 1e-1)
}

scheduler = MedianStoppingRule(metric='average_test_score',
                        grace_period=10.0)

tune_search = TuneRandomizedSearchCV(clf,
            param_distributions,
            scheduler=scheduler,
            early_stopping=True,
            n_iter=3,
            iters=10
            )
tune_search.fit(x_train, y_train)

pred = tune_search.predict(x_test)
accuracy = np.count_nonzero(np.array(pred) == np.array(y_test))/len(pred)
print(accuracy)
