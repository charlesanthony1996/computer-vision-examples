#simple base estimator for linear regression only using numpy

import numpy as np
import matplotlib

class LinearRegression:
    def __init__(self, learning_rate = 0.01, n_iterations= 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    
    def fit(self,x, y):
        n_samples , n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self._iterations):
            y_predicted = np.dot(x, self.weights) + self.bias
            dw = (1/ n_samples) * np.dot(x.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    
    def predict(self, x):
        y_predicted = np.dot(x, self.weights) + self.bias
        return y_predicted



import numpy as np

# generate random data
np.random.seed(0)
n_samples = 100
x = np.linspace(0, 10, n_samples).reshape(n_samples, 1)
y = x * 2 + np.random.randn(n_samples, 1)

# initialize and fit the model
model = LinearRegression()
model.fit(x, y)


# predict new values
x_new = 
