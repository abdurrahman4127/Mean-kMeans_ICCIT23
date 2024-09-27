import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

class ElbowMethod:
    def __init__(self, data, max_clusters):
        self.data = data
        self.max_clusters = max_clusters
        self.inertia_values = []

    def calculate_inertia(self):
        for k in range(1, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=k, n_init=100)
            kmeans.fit(self.data)
            self.inertia_values.append(kmeans.inertia_)

    def plot_elbow_curve(self):
        plt.plot(range(1, self.max_clusters + 1), self.inertia_values, marker='o')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    elbow = ElbowMethod(X, max_clusters=10)
    elbow.calculate_inertia()
    elbow.plot_elbow_curve()