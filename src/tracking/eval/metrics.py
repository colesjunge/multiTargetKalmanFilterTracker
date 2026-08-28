import numpy as np
from scipy import linalg


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

def nees(x_est: np.ndarray, P: np.ndarray, x_true: np.ndarray) -> float:
    """Calculate the Normalized Estimation Error Squared (NEES).
    In: x_est shape (n,), P shape (n, n), x_true shape (n,)
    Out: scalar
    """

    if x_est.shape != x_true.shape:
        raise ValueError("Estimated state and true state must have the same shape.")

    if P.shape[0] != P.shape[1] or P.shape[0] != x_est.shape[0]:
        raise ValueError("Covariance matrix P must be square and match the dimension of the state.")

    # Calculate the estimation error
    error = x_est - x_true

    # Calculate NEES
    y = linalg.solve(P, error)
    nees_value = error @ y

    return float(nees_value)


def average_nees(snapshots: list[tuple[np.ndarray, np.ndarray]], truths: list[np.ndarray]) -> float:
    """
    Calculate the mean NEES over entire run
    In: snapshots; list of (x_est, P) tuples, one entry per (run, timestep) pair across the entire Monte Carlo simulation
        truths; list of x_true vectors, same length as snapshots, each (n,)
    Out: mean NEES over every entry; scalar
    """
    avg_nees = 0

    for i in range(len(snapshots)):
        avg_nees += nees(snapshots[i][0], snapshots[i][1], truths[i])

    return float(avg_nees / len(snapshots))


def peak_position_error(estimates: np.ndarray, truth: np.ndarray) -> tuple[float, int]:
    """Calculate the peak error between estimated and true positions.
    In: estimates shape (K, 2), truth shape (K, 2)
    Out: result (peak_error, peak_error_frame)
    """

    if estimates.shape != truth.shape:
        raise ValueError("Estimates and truth must have the same shape.")

    # Difference between estimated and true positions
    diff = estimates - truth
    squared_diff = np.linalg.norm(diff, axis=1)

    # Peak error
    peak_error = np.max(squared_diff)

    # Get index for peak error
    peak_error_frame = np.argmax(squared_diff)

    return (float(peak_error), int(peak_error_frame))
