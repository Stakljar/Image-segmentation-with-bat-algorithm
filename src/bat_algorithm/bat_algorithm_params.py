import numpy.typing as npt
import numpy as np

class BatAlgorithmParams:
    def __init__(self, total_bats: int, num_iterations: int, dimension: int,
                 min_position: float | list[float] | npt.NDArray[np.float64],
                 max_position: float | list[float] | npt.NDArray[np.float64],
                 min_frequency: float=0., max_frequency: float=1., sigma: float=0.1,
                 gamma: float=0.1, alpha: float=0.97) -> None:
        if(total_bats < 1):
            raise ValueError("Argument total_bats cannot be less than 1")
        if(dimension < 1):
            raise ValueError("Argument dimension cannot be less than 1")
        if(isinstance(min_position, (list, np.ndarray))):
            if(len(min_position) != dimension and len(min_position) != 1):
                raise ValueError("Argument min_position must have same size as dimension parameter's value or have a single value")
        if(isinstance(max_position, (list, np.ndarray))):
            if(len(max_position) != dimension and len(max_position) != 1):
                raise ValueError("Argument max_position must have same size as dimension parameter's value or have a single value")
        self.total_bats: int = total_bats
        self.num_iterations: int = num_iterations
        self.dimension: int = dimension
        self.min_position: npt.NDArray[np.float64] = np.copy(min_position)
        self.max_position: npt.NDArray[np.float64] = np.copy(max_position)
        self.min_frequency: float = min_frequency
        self.max_frequency: float = max_frequency

        self.sigma: float = sigma
        self.gamma: float = gamma
        self.alpha: float = alpha
