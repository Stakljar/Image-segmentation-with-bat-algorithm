from bat_algorithm.bat import Bat
import numpy as np
from methods.kmeans import KMeans
from bat_algorithm.bat_algorithm_params import BatAlgorithmParams
import numpy.typing as npt

class BatAlgorithmWithKMeans(BatAlgorithmParams):
    def __init__(self, kmeans: KMeans, total_bats: int, num_iterations: int,
                 min_position: float | list[float] | npt.NDArray[np.float64],
                 max_position: float | list[float] | npt.NDArray[np.float64],
                 dimension: int, min_frequency: float=0., max_frequency: float=1.,
                 sigma: float=0.1, gamma: float=0.1, alpha: float=0.97) -> None:

        super().__init__(total_bats=total_bats, num_iterations=num_iterations, dimension=dimension,
                         min_position=min_position, max_position=max_position, min_frequency=min_frequency,
                         max_frequency=max_frequency, sigma=sigma, gamma=gamma, alpha=alpha)
        self.kmeans: KMeans = kmeans
        self.space_dimension: int = kmeans.data.shape[1]

    def _initialize_population(self) -> tuple[list[Bat], list[npt.NDArray[np.int64]]]:
        bats: list[Bat] = []
        labels_list: list[npt.NDArray[np.int64]] = []
        for _ in range(self.total_bats):
            position, labels, inertia = self.kmeans.run(self.dimension)
            velocity: npt.NDArray[np.float64] = np.zeros((self.dimension, self.space_dimension))
            frequency: float = 0.
            marginal_pulse_rate: float = 1.
            pulse_rate: float = 0.
            loudness: float = 1.
            bats.append(Bat(
                position=position,
                velocity=velocity,
                frequency=frequency,
                fitness=inertia,
                marginal_pulse_rate=marginal_pulse_rate,
                pulse_rate=pulse_rate,
                loudness=loudness,
            ))
            labels_list.append(labels)
        return bats, labels_list
    
    def run(self) -> tuple[npt.NDArray[np.float64], float, npt.NDArray[np.int64]]:
        bats, labels_list = self._initialize_population()
        t: int = 0
        best_index, best_bat = min(enumerate(bats), key=lambda x: x[1].fitness)
        best_labels: npt.NDArray[np.int64] = labels_list[best_index]
        while(t < self.num_iterations):
            t = t + 1
            for bat in bats:
                bat.frequency = np.random.uniform(self.min_frequency, self.max_frequency)
                bat.velocity = bat.velocity + (bat.position-best_bat.position) * bat.frequency
                temp_position: npt.NDArray[np.float64] = bat.position + bat.velocity
                if(np.random.rand() > bat.pulse_rate):
                    temp_position = best_bat.position + self.sigma * \
                        np.random.normal(0, 1, self.dimension)[:, np.newaxis] * np.mean([bat.loudness for bat in bats])
                temp_position = np.clip(temp_position, self.min_position, self.max_position)
                labels: npt.NDArray[np.int64] = self.kmeans.assign_clusters(temp_position)
                new_fitness: float = self.kmeans.calculate_inertia(temp_position, labels)
                if(np.random.rand() < bat.loudness and new_fitness < bat.fitness):
                    bat.position = temp_position
                    bat.fitness = new_fitness
                    bat.pulse_rate = bat.marginal_pulse_rate * (1 - np.exp(-self.gamma*t))
                    bat.loudness = self.alpha * bat.loudness
                    if(new_fitness < best_bat.fitness):
                        best_labels = labels
                        best_bat = bat
        return best_bat.position, best_bat.fitness, best_labels
