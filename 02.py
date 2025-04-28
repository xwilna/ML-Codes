import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score

mnist = fetch_openml("cifar10")
X,y = mnist.data, mnist.target

svm_model1 = SVC(probability=True)
svm_model1.fit(X, y)
y_pred_proba = svm_model1.predict_proba(X)


svm_model2 = SVC()
svm_model2.fit(y_pred_proba, y)

y_pred = svm_model2.predict(X)



