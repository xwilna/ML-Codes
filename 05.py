import numpy as np
from cuml.svm import SVC as cuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

class OptimizedStackedModel:
    def __init__(self, n_neighbors=3, n_components=50):
        self.n_neighbors = n_neighbors
        self.n_components = n_components

        self.svm = cuSVC(kernel='rbf', C=10, probability=True)
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, n_jobs=-1)
        self.lr = LogisticRegression(max_iter=500, solver='liblinear', n_jobs=-1)

        meta_model = LogisticRegression(max_iter=500, solver='liblinear')

        self.stacking_clf = StackingClassifier(
            estimators=[('svm', self.svm), ('knn', self.knn), ('lr', self.lr)],
            final_estimator=meta_model
        )

        self.pca = PCA(n_components=self.n_components)

    def preprocess_data(self, X):
        X_flat = X.reshape(X.shape[0], -1) 
        X_pca = self.pca.fit_transform(X_flat)  
        return X_pca

    def fit(self, X_train, y_train):
        X_train_pca = self.preprocess_data(X_train)
        self.stacking_clf.fit(X_train_pca, y_train)

    def predict(self, X_test):
        X_test_pca = self.pca.transform(X_test.reshape(X_test.shape[0], -1))
        return self.stacking_clf.predict(X_test_pca)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")

cifar10 = fetch_openml('CIFAR_10', version=1)
X = np.array(cifar10.data, dtype=np.float32)
y = np.array(cifar10.target, dtype=np.int32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = OptimizedStackedModel(n_neighbors=1, n_components=50)
model.fit(X_train, y_train)
model.evaluate(X_test, y_test)
