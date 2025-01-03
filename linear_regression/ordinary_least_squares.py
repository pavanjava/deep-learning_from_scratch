from typing import List


# Define the LinearRegression class with methods to fit and predict.
class LinearRegression:
    def __init__(self, data_set: List):
        self.data = data_set
        self.alpha = 0
        self.beta = 0

    def fit(self):
        x_sum = 0
        y_sum = 0
        for _tuple in data_set:
            x_sum += _tuple[0]
            y_sum += _tuple[1]

        x_mean = x_sum / len(data_set)  # calculate mean of x values
        y_mean = y_sum / len(data_set)  # calculate mean of y values

        covariance_xy = 0
        variance_x = 0
        for _tuple in data_set:
            covariance_xy += ((_tuple[0] - x_mean) * (_tuple[1] - y_mean))  # calculate covariance of x and y
            variance_x += (_tuple[0] - x_mean) ** 2  # calculate variance of x

        self.alpha = covariance_xy / variance_x  # calculate slope (alpha)
        self.beta = y_mean - self.alpha * x_mean  # calculate intercept (beta)

    # predict the value of y for a given x value using the linear regression model.
    def predict(self, data: int) -> float:
        return (self.alpha * data) + self.beta


# driver code to test the above class
data_set = [(1, 3), (2, 4.5), (3, 6), (4, 7.5), (5, 8)]
linear_regression = LinearRegression(data_set=data_set)
linear_regression.fit()
print(linear_regression.predict(data=10))
