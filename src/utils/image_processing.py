import numpy as np


def sliding_window_iterator(image: np.ndarray, window_size: int,
                            step_size: int = None):

    step_size = step_size or window_size

    for row in range(0, image.shape[0], step_size):
        for col in range(0, image.shape[1], step_size):
            window = image[row: row + window_size,
                           col: col + window_size]

            yield (row, col, window)
