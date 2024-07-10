import numpy as np
import numpy.typing as npt
import torch

def segment_image_by_thresholds(image: npt.NDArray[np.uint8], thresholds: list[np.float64] | npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    sorted_thresholds = np.sort(thresholds)
    segmented_image = np.copy(image)
    for i in range(sorted_thresholds.size + 1):
        condition = (
            (segmented_image <= (sorted_thresholds[i] if i < sorted_thresholds.size else 255)) & 
            (segmented_image > (sorted_thresholds[i-1] if i > 0 else -1))
        )
        if(segmented_image[condition].size > 0):
            segmented_image[condition] = np.mean(segmented_image[condition])
                                       # np.bincount(segmented_image[condition]).argmax()
    return segmented_image

def flatten_rows_and_cols_of_image(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    if(image.ndim == 3):
        pixels = image.reshape(-1, 3)
    else:
        pixels = image.reshape(-1, 1)
    return pixels

def segment_image_by_clusters(image: npt.NDArray[np.uint8], centroids: npt.NDArray[np.float64] | list[float], labels: npt.NDArray[np.int64] | list[float]):
    return np.round(centroids).astype(np.uint8)[labels].reshape(image.shape)

def get_pascal_voc_classes_colors() -> npt.NDArray[np.uint8]:
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors_tensor = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors_tensor % 255).numpy().astype("uint8")
    return colors
