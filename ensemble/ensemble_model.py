from typing import List

import numpy as np


def average_probabilities(predictions: List[np.ndarray]) -> np.ndarray:
    if not predictions:
        raise ValueError("No predictions provided for ensembling")
    stacked = np.stack(predictions, axis=0)
    avg = stacked.mean(axis=0)
    return avg


__all__ = ["average_probabilities"]



