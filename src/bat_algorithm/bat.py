import numpy as np
import numpy.typing as npt

class Bat:
    def __init__(self, position: npt.NDArray[np.float64], velocity: npt.NDArray[np.float64],
                 frequency: float, fitness: float, pulse_rate: float=0., marginal_pulse_rate: float=1.,
                 loudness: float=0.) -> None:
        self.position: npt.NDArray[np.float64] = position
        self.velocity: npt.NDArray[np.float64] = velocity
        self.frequency: float = frequency
        self.marginal_pulse_rate: float = marginal_pulse_rate
        self.pulse_rate: float = pulse_rate
        self.loudness: float = loudness
        self.fitness: float = fitness
