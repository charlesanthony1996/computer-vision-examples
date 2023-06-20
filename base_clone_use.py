from sklearn.base import clone
from sklearn.datasets import load_iris

# from sklearn.base import clone
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier

# # Create a list of estimators
# estimators = [LogisticRegression(), RandomForestClassifier()]

# # Clone the estimators
# cloned_estimators = clone(estimators, safe=True)

# for i, estimator in enumerate(cloned_estimators):
#     print(f"Original estimator type: {type(estimators[i])}")
#     print(f"Cloned estimator type: {type(estimator)}")
#     print("\n")

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# load iris dataset
iris = load_iris()
x, y = iris.data, iris.target

# split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

# initialize a decision tree classifier
clf = DecisionTreeClassifier(max_depth = 3, random_state=42)

# fit on training data
clf.fit(x_train, y_train)


# clone the trained model
clf_clone = clone(clf)
# print(clf_clone)

# even after cloning, clf clone should not have been fitted yet
try:
    clf_clone.predict(x_test)
except Exception as e:
    print(f"cloned model error: (as expected) as: {e}")


clf_clone.fit(x_train, y_train)

assert accuracy_score(y_test, clf.predict(x_test)) == accuracy_score(y_test, clf_clone.predict(x_test)),  "The models dont match!"

print("both models matched")




