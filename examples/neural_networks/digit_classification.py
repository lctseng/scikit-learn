#!/usr/bin/env python3
import time

from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

import pdb

TRAIN_DATA_SIZE = 60000

print("Downloading test data from mldata.org")
mnist = fetch_mldata("MNIST original")
print("Data downloaded")

# rescale the data, use the traditional train/test split
X = mnist.data / 255.
y = mnist.target.astype(X.dtype)






X_train, X_test = X[:TRAIN_DATA_SIZE], X[TRAIN_DATA_SIZE:]
y_train, y_test = y[:TRAIN_DATA_SIZE], y[TRAIN_DATA_SIZE:]

print("Training data size: %d" % len(X_train))
print("Test data size: %d" % len(X_test))


mlp = MLPClassifier(hidden_layer_sizes=(200,200), max_iter=5, alpha=1e-4,
                    solver='adam', verbose=True, tol=1e-4, random_state=1,
                    batch_size=200, learning_rate_init=0.005,
                    useCuda=True)

ts = time.time()
mlp.fit(X_train, y_train)
print("Fit time: %f" % (time.time() - ts))

"""
y_predicted = mlp.predict(X_test)


print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))


print("Confusion matrix:\n%s"
    % metrics.confusion_matrix(y_test, y_predicted))
print("Classification report:\n%s\n"
    % metrics.classification_report(y_test, y_predicted))
"""