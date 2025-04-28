from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC


class BoostedSVC(BaseEstimator, ClassifierMixin):
    def __init__(self, C_1=1, kernel_1='rbf', C_2=1, kernel_2='rbf'):
        self.svm_model1 = SVC(C=C_1, kernel=kernel_1, probability=True)
        self.svm_model2 = SVC(C=C_2, kernel=kernel_2)

    def fit(self, X, y, sample_weight=None):
        self.svm_model1.fit(X, y, sample_weight=sample_weight)
        y_pred_proba = self.svm_model1.predict_proba(X)
        self.svm_model2.fit(y_pred_proba, y, sample_weight=sample_weight)

    def predict(self, X):
        y_pred_proba = self.svm_model1.predict_proba(X)
        self.svm_model2.predict(y_pred_proba)

    def score(self, X, y):
        pass

    def plot(self):
        pass

    def outlier_detection(self, X):
        pass


param_grid = {"C_1": [0, 0.01, 0.1, 0.5, 1, 10],
              "kernel_1": ['linear', 'rbf'],
              "C_2": [0, 0.01, 0.1, 0.5, 1, 10],
              "kernel_2": ['linear', 'rbf']
              }

# rnd_model = RandomizedSearchCV(BoostedSVC(), param_grid, cv=5, scoring='accuracy', verbose=2)

# grid_model = GridSearchCV(BoostedSVC(), param_grid, cv=5, scoring='accuracy', verbose=2)
# grid_model.fit(X,y)
