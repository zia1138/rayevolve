import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3

    np.random.seed(42)
    points = np.random.randn(n, d)

    return points
