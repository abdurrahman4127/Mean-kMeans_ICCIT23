import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

class NormalKMeans:
    def __init__(self, X, max_iters, max_clusters, k=None):
        self.max_iters = max_iters
        self.data = X
        self.k = k if k is not None else self.find_best_k(max_clusters)
        self.kmeans = None
        self.labels = None
        self.centers = None
        self.iterations = None

    def find_best_k(self, max_clusters):
        inertia_values = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, n_init=self.max_iters)
            kmeans.fit(self.data)
            inertia_values.append(kmeans.inertia_)
        diff_inertia = [inertia_values[i] - inertia_values[i + 1] for i in range(len(inertia_values) - 1)]
        return diff_inertia.index(max(diff_inertia)) + 2

    def fit(self):
        self.kmeans = KMeans(n_clusters=self.k, n_init=self.max_iters)
        self.kmeans.fit(self.data)
        self.labels = self.kmeans.labels_
        self.centers = self.kmeans.cluster_centers_
        self.iterations = self.kmeans.n_iter_

    def visualize(self):
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, cmap='viridis')
        plt.scatter(self.centers[:, 0], self.centers[:, 1], marker='X', s=200, c='red')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('K-means Clustering')
        plt.show()

    def silhouette_score(self):
        return silhouette_score(self.data, self.labels)

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data[:, :2] 
    normalkmeans = NormalKMeans(X, max_iters=100, max_clusters=10, k=2)
    normalkmeans.fit()
    normalkmeans.visualize()
    print("Silhouette Score:", normalkmeans.silhouette_score())
    print("Iterations:", normalkmeans.iterations)