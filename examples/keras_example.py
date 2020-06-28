"""
An example training a Keras model, performing
grid search using TuneGridSearchCV.
"""

from keras.datasets import mnist
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from tune_sklearn import TuneGridSearchCV

nb_classes = 10
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train[:500]
y_train = y_train[:500]
X_test = X_test[:100]
y_test = y_test[:100]

X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


def create_model(optimizer="rmsprop", kernel_initializer="glorot_uniform"):
    model = Sequential()
    model.add(Dense(512, input_shape=(784, )))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(512, kernel_initializer=kernel_initializer))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(10, kernel_initializer=kernel_initializer))
    model.add(Activation("softmax"))  # This special "softmax" a
    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


model = KerasClassifier(build_fn=create_model)
optimizers = ["rmsprop", "adam"]
kernel_initializer = ["glorot_uniform", "normal"]
epochs = [5, 10]
param_grid = dict(
    optimizer=optimizers,
    nb_epoch=epochs,
    kernel_initializer=kernel_initializer)
grid = TuneGridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_train, Y_train)
print(grid_result.best_params_)
print(grid_result.cv_results_)
