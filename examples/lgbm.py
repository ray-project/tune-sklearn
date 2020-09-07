"""Example using LightGBM, performing randomized search with TuneSearchCV.

Example taken from https://mlfromscratch.com/gridsearch-keras-sklearn/#/
"""

import lightgbm as lgb
from tune_sklearn import TuneSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load breast cancer dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = lgb.LGBMClassifier()
param_dists = {
    "n_estimators": [400, 700, 1000],
    "colsample_bytree": [0.7, 0.8],
    "max_depth": [15, 20, 25],
    "num_leaves": [50, 100, 200],
    "reg_alpha": [1.1, 1.2, 1.3],
    "reg_lambda": [1.1, 1.2, 1.3],
    "min_split_gain": [0.3, 0.4],
    "subsample": [0.7, 0.8, 0.9],
    "subsample_freq": [20]
}

gs = TuneSearchCV(model, param_dists, n_trials=5, scoring="accuracy")
gs.fit(X_train, y_train)
print(gs.cv_results_)

pred = gs.predict(X_test)
correct = 0
for i in range(len(y_test)):
    if pred[i] == y_test[i]:
        correct += 1
print("Accuracy:", correct / len(pred))
