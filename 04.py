from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

mnist = fetch_openml("mnist_784")
X, y = mnist.data, mnist.target

pip_line = Pipeline(
    [('scale', StandardScaler()),
     ('svm', SVC())
     ]
)

pip_line.fit(X, y)
