import numpy as np
from typing import Any


# Define a class for Linear Regression using Gradient Descent
class LinearRegression:
    def __init__(self, X: Any, y: Any, alpha: float = 0.0, beta: float = 0.0, epochs: int = 1000, lr: float = 0.01,
                 is_print: bool = False):
        self.alpha = alpha
        self.beta = beta
        self.epochs = epochs
        self.learning_rate = lr
        self.X = np.array(X)
        self.y = np.array(y)
        self.is_print = is_print

    def fit(self):
        for _ in range(self.epochs):
            y_pred = self.alpha * self.X + self.beta
            mse = np.sum((self.y - y_pred) ** 2).mean()
            alpha_d = -(2 / len(self.X)) * np.sum(self.X * (self.y - y_pred))
            beta_d = -(2 / len(self.X)) * np.sum(self.y - y_pred)
            self.alpha -= self.learning_rate * alpha_d
            self.beta -= self.learning_rate * beta_d
            if self.is_print:
                print("cost {}, alpha {}, beta {}".format(mse, self.alpha, self.beta))

    def predict(self, x_):
        return self.alpha * x_ + self.beta


x = [1, 2, 3, 4, 5]
y = [5, 7, 9, 11, 13]

l = LinearRegression(X=x, y=y, epochs=10000, lr=0.001)
l.fit()
print(l.predict(10))
