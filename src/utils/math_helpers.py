import numpy as np


def get_closest_value(target, array):
    if len(array) == 0:
        return None
    idx = np.searchsorted(array, target)
    if idx == 0:
        return float(array[0])
    if idx == len(array):
        return float(array[-1])
    left, right = float(array[idx - 1]), float(array[idx])
    return left if abs(target - left) < abs(target - right) else right
