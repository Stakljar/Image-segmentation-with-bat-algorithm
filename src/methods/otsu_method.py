import numpy as np
import numpy.typing as npt

class OtsuMethod:
    def __init__(self, image: npt.NDArray[np.uint8]) -> None:
        self.image: npt.NDArray[np.uint8] = np.copy(image)

    def calculate_within_class_variance(self, thresholds: npt.NDArray[np.float64] | list[float]) -> float:
        within_class_variance: float = 0.
        sorted_thresholds: npt.NDArray[np.float64] = np.sort(thresholds)
        for i in range(sorted_thresholds.size + 1):
            segment = self.image[(self.image <= (sorted_thresholds[i] if i < sorted_thresholds.size else 255))
                                 & (self.image > (sorted_thresholds[i-1] if i > 0 else -1))]
            segment_occurrence_probability: float = segment.size / self.image.size
            segment_variance: float = np.var(segment) if segment.size > 0 else 0.
            within_class_variance += segment_occurrence_probability * segment_variance
        return within_class_variance
