import numpy as np


def position_rmse(estimates: np.ndarray, truth: np.ndarray) -> float:
    """Calculate the root mean square error (RMSE) between estimated and true positions.
    In: estimates shape (K, 2), truth shape (K, 2)
    Out: scalar.
    """

    if estimates.shape != truth.shape:
        raise ValueError("Estimates and truth must have the same shape.")

    # Difference between estimated and true positions
    diff = estimates - truth

    # Squared Euclidean distance at each timestep
    squared_distance = np.sum(diff ** 2, axis=1)

    # Average squared distance across all K timesteps
    mean_squared_distance = np.mean(squared_distance)

    # Take the square root
    rmse = np.sqrt(mean_squared_distance)

    return float(rmse)
