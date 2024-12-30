import math

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


# create synthetic data for logistic regression
def create_data() -> Tuple[float, float]:
    x, y = make_classification(n_samples=1000, n_features=2, n_classes=2, n_clusters_per_class=1, random_state=22,
                               class_sep=20, n_informative=1, n_redundant=0, hypercube=False)
    return x, y


# create synthetic data for logistic regression
x, y = create_data()
# plt.figure(figsize=(12,8))
# plt.scatter(x[:,0], x[:,1], c=y, cmap="winter", s=100)
# plt.show()

# print(x)
# print(y)

# split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

epochs = 1000
lr = 0.01
w = np.zeros(x_train[0].shape)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(w.shape)


# define sigmoid function
def compute_sigmoid(val):
    return 1 / (1 + np.exp(-val))


def gradient(weights=None):
    for _ in range(epochs):
        predictions = compute_sigmoid(np.dot(weights, x_train.T))
        error = y_train - predictions
        weights += lr * np.dot(error, x_train) / len(y_train)
        # print(f"Epoch {_}, Loss: {np.mean(np.square(error))}")
    return weights[1:], weights[0]


print(gradient(w))
