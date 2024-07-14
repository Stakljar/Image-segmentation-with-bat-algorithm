from bat_algorithm.bat import Bat
import numpy as np
from typing import Callable
import numpy.typing as npt
from bat_algorithm.bat_algorithm_params import BatAlgorithmParams

class BatAlgorithm(BatAlgorithmParams):
    def __init__(self, total_bats: int, num_iterations: int, dimension: int,
                 min_position: float | list[float] | npt.NDArray[np.float64],
                 max_position: float | list[float] | npt.NDArray[np.float64],
                 min_frequency: float=0., max_frequency: float=1., sigma: float=0.1,
                 gamma: float=0.1, alpha: float=0.97) -> None:

        super().__init__(total_bats=total_bats, num_iterations=num_iterations,
                         dimension=dimension, min_position=min_position, max_position=max_position,
                         min_frequency=min_frequency, max_frequency=max_frequency, sigma=sigma,
                         gamma=gamma, alpha=alpha)

    def _initialize_population(self, objective_function: Callable[[npt.NDArray[np.float64]], float]) -> list[Bat]:
        bats: list[Bat] = []
        for _ in range(self.total_bats):
            position: npt.NDArray[np.float64] = np.random.uniform(self.min_position, self.max_position, self.dimension)
            velocity: npt.NDArray[np.float64] = np.zeros(self.dimension)
            frequency: float = 0.
            marginal_pulse_rate: float = 1.
            pulse_rate: float = 0.
            loudness: float = 1.
            fitness: float = objective_function(position)
            bats.append(Bat(
                position=position,
                velocity=velocity,
                frequency=frequency,
                fitness=fitness,
                marginal_pulse_rate=marginal_pulse_rate,
                pulse_rate=pulse_rate,
                loudness=loudness,
            ))
        return bats

    def run(self, objective_function: Callable[[npt.NDArray[np.float64]], float]) -> tuple[npt.NDArray[np.float64], float]:
        bats: list[Bat] = self._initialize_population(objective_function)
        t: int = 0
        best_bat: Bat = min(bats, key=lambda bat: bat.fitness)
        while(t < self.num_iterations):
            t = t + 1
            for bat in bats:
                bat.frequency = np.random.uniform(self.min_frequency, self.max_frequency)
                bat.velocity = bat.velocity + (bat.position-best_bat.position) * bat.frequency
                temp_position: npt.NDArray[np.float64] = bat.position + bat.velocity
                if(np.random.rand() > bat.pulse_rate):
                    temp_position = best_bat.position + self.sigma * np.random.normal(0, 1, self.dimension)\
                    * np.mean([bat.loudness for bat in bats])
                temp_position = np.clip(temp_position, self.min_position, self.max_position)
                new_fitness: float = objective_function(temp_position)
                if(np.random.rand() < bat.loudness and new_fitness < bat.fitness):
                    bat.position = temp_position
                    bat.fitness = new_fitness
                    bat.pulse_rate = bat.marginal_pulse_rate * (1 - np.exp(-self.gamma*t))
                    bat.loudness = self.alpha * bat.loudness
                    if(new_fitness < best_bat.fitness):
                        best_bat = bat
        return best_bat.position, best_bat.fitness
