import numpy as np
import numpy.typing as npt

class KMeans:
    def __init__(self, data: npt.NDArray[np.float64], max_iterations: int=10, tolerance: float=5e-8) -> None:
        self.data: npt.NDArray[np.float64] = np.copy(data)
        self.max_iterations: int = max_iterations
        self.tolerance: float = tolerance

    def run(self, k: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], float]:
        unique_data: npt.NDArray[np.float64] = np.unique(self.data, axis=0)
        centroids: npt.NDArray[np.float64] = unique_data[np.random.choice(unique_data.shape[0], size=k, replace=False)]
        labels: npt.NDArray[np.int64] = self.assign_clusters(centroids)
        for _ in range(self.max_iterations):
            new_centroids: npt.NDArray[np.float64] = np.array([np.mean(self.data[labels == label], axis=0) for label in np.unique(labels)])
            if(new_centroids.shape != centroids.shape):
                break
            if(np.all(np.abs(new_centroids - centroids) <= self.tolerance)):
                break
            centroids = new_centroids
            labels = self.assign_clusters(centroids)
        inertia: float = self.calculate_inertia(centroids, labels)
        return centroids, labels, inertia

    def run_multiple_times(self, k: int, attempts: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], float]:
        best_inertia: float = np.inf
        best_centroids: npt.NDArray[np.float64] = np.array([])
        best_labels: npt.NDArray[np.int64] = np.array([])
        for _ in range(attempts):
            centroids, labels, inertia = self.run(k)
            if(inertia < best_inertia):
                best_centroids = centroids
                best_labels = labels
                best_inertia = inertia
        return best_centroids, best_labels, best_inertia

    def assign_clusters(self, centroids: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        distances: npt.NDArray[np.float64] = np.linalg.norm(self.data[:, np.newaxis, :] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def calculate_inertia(self, centroids: npt.NDArray[np.float64], labels: npt.NDArray[np.int64]) -> float:
        inertia: float = 0.
        for label, centroid in enumerate(centroids):
            inertia += np.sum(np.linalg.norm(self.data[labels == label] - centroid, axis=1) ** 2)
        return inertia
