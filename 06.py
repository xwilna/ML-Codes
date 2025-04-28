import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score

# Ensemble
# Boosting

mnist = fetch_openml("mnist_784")
X,y = mnist.data, mnist.target

n_samples = len(X)

# Model 1
logistic_model = LogisticRegression()
logistic_model.fit(X, y)
y_pred = logistic_model.predict(X)


# Calculate Sample Weights
sample_weights = np.ones_like(y) / n_samples
error = np.sum(sample_weights * (y_pred != y)) / np.sum(sample_weights)

alpha = 0.5 * np.log((1-error) / np.maximum(error, 1e-10))

sample_weights *= np.exp(alpha * (y_pred != y))
sample_weights /= np.sum(sample_weights)

# Model 2
svm_model = SVC()
svm_model.fit(X, y, sample_weight=sample_weights)

