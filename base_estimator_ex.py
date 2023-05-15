import numpy as np
from sklearn.base import BaseEstimator

class MyEstimator(BaseEstimator):
    def __init__(self):
        self.coef_ = None

    def fit(self, x, y):
        # train the model on the data x and target values y
        self.coef_ = np.mean(y) / np.mean(x, axis=0)


    def predict(self, x):
        # use the trained model to make predictions on new data x
        return np.dot(x, self.coef_)


# create an instance of the custom estimator
estimator = MyEstimator()

# fit the model to some training data
x_train = np.array([[1, 2, 3], [4, 5, 6]])
y_train = np.array([0, 1])
estimator.fit(x_train, y_train)


# make preds on some test data
x_test = np.array([[7, 8, 9], [10, 11, 12]])
y_pred = estimator.predict(x_test)

print(y_pred)

