from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=20)

plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1])
plt.show()
