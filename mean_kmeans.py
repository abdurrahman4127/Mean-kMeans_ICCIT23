import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

class MeanKMeans:
    def __init__(self, X, max_iters, max_clusters, k=None):
        self.X = X
        self.k = k if k is not None else self.find_best_k(max_clusters)
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None
        self.iterations = 0

    def find_best_k(self, max_clusters):
        inertia_values = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, n_init=100)
            kmeans.fit(self.X)
            inertia_values.append(kmeans.inertia_)
        diff_inertia = [inertia_values[i] - inertia_values[i + 1] for i in range(len(inertia_values) - 1)]
        return diff_inertia.index(max(diff_inertia)) + 2

    def mean_of_clusters(self, labels):
        mean_list = []
        for i in range(self.k):
            cluster_points = self.X[labels == i]
            mean_list.append(cluster_points.mean(axis=0) if len(cluster_points) > 0 else np.zeros(self.X.shape[1]))
        return np.array(mean_list)

    def fit(self):
        np.random.seed(42)
        self.centroids = self.X[np.random.choice(self.X.shape[0], self.k, replace=False)]

        for _ in range(self.max_iters):
            self.iterations += 1
            distances = np.linalg.norm(self.X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)
            new_centroids = self.mean_of_clusters(self.labels)

            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

    def visualize(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.labels, cmap='viridis')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='X', s=200, c='red')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Mean-KMeans Clustering')
        plt.show()

    def silhouette_score(self):
        return silhouette_score(self.X, self.labels)

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data[:, :2]  
    meankmeans = MeanKMeans(X, max_iters=100, max_clusters=10, k=None)
    meankmeans.fit()
    meankmeans.visualize()
    print("Silhouette Score:", meankmeans.silhouette_score())
    print("Iterations:", meankmeans.iterations)