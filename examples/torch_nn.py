"""
An example training a PyTorch NeuralNetClassifier, performing
grid search using TuneGridSearchCV. 
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from tune_sklearn.tune_search import TuneGridSearchCV
X, y = make_classification(1000, 20, n_informative=10, random_state=0)
X = X.astype(np.float32)
y = y.astype(np.int64)
class MySimpleModule(nn.Module):
    def __init__(self, num_units=10, nonlin=F.relu):
        super(MySimpleModule, self).__init__()
        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 2)
    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X))
        return X
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)

net = NeuralNetClassifier(
    MySimpleModule,
    max_epochs=10,
    lr=0.1,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
)

params = {
    'lr': [0.01, 0.02],
    'max_epochs': [10, 20],
    'module__num_units': [10, 20],
}

gs = TuneGridSearchCV(net, params, refit=False, scoring='accuracy')
# import ipdb; ipdb.set_trace()
gs.fit(X, y)
print(gs.best_score_, gs.best_params_)