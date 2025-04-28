import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)
mean = [0, 0, 0]
cov = [[3, 1, 1],
       [1, 2, 0.5],
       [1, 0.5, 1]]
X = np.random.multivariate_normal(mean, cov, 200)

pca = PCA(n_components=3)
pca.fit(X)
components = pca.components_     
explained = pca.explained_variance_ 


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.5)


origin = np.mean(X, axis=0)
for i in range(3):
    vec = components[i] * np.sqrt(explained[i]) * 3
    ax.quiver(*origin, *vec, color=['r','g','b'][i], linewidth=3)


ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()