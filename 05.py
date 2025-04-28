import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score

mnist = fetch_openml("mnist_784")
X,y = mnist.data, mnist.target

# Base Model
svm_model1 = SVC(probability=True)
svm_model1.fit(X, y)
y_pred_proba_1 = svm_model1.predict_proba(X)

knn_model2 = KNeighborsClassifier()
knn_model2.fit(X, y)
y_pred_proba_2 = knn_model2.predict_proba(X)

logistic_regression_model1 = LogisticRegression()
logistic_regression_model1.fit(X, y)
y_pred_proba_3 = logistic_regression_model1.predict_proba(X)

# Meta Model
logistic_model = LogisticRegression()
X = np.c_[y_pred_proba_1, y_pred_proba_2, y_pred_proba_3]
logistic_model.fit(X, y)
y_pred = logistic_model.predict(X)



