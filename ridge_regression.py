import sklearn
from sklearn.linear_model import Ridge

class RidgeModel:
    def __init__(self, alpha=0):
        self.clf = Ridge(alpha=alpha)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
