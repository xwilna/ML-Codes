import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.utils import compute_class_weight



# Ensemble
# Boosting

mnist = fetch_openml("mnist_784")
X,y = mnist.data, mnist.target

# Model 1
logistic_model = LogisticRegression()
logistic_model.fit(X, y)
y_pred = logistic_model.predict(X)

wrong_ = y[y != y_pred]
print(wrong_)
print(len(wrong_))
print(accuracy_score(y, y_pred))

# Predict 1
# sample_weight = np.ones_like(y)

# correct_ = y[y == y_pred]  --> weight 1
# wrong_ = y[y != y_pred]    --> weight 2
# sample_weight[y!=y_pred] *= 1.2


# Model 2
# svm_model = SVC(class_weight=compute_class_weight('balanced', np.unique(y), y))   # Dict --> len(unique(y))
# svm_model.fit(X, y, sample_weight=sample_weight)                                  # List --> len(X)


# outlier
# 10
# 2*10 == 20
# overfit