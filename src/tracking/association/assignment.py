import numpy as np
from scipy import optimize



def hungarian_assign(cost: np.ndarray) -> tuple[list[tuple[int,int]], list[int], list[int]]:
    """ Solves assignment problem using the Hungarian algorithm
    Rejects any match whose oroginal cost exceeded the gate with inf entries denied

    In: cost shape (M, N)
    Out: (matches, unmatched_tracks, unmatched_detections)
    matches; list of (track_idx, det_idx) tuples; globally optimal assignments that passed the gate
    unmatched_tracks; list of track indices with no viable detection
    unmatched_detections; list of detection indices with no viable track
    """

    matches = []
    unmatched_tracks = []
    unmatched_detections = []

    # Use large number instead of inf for solver
    sentinel_cost = cost.copy()
    sentinel_cost[np.isinf(sentinel_cost)] = 1e6

    row_ind, col_ind = optimize.linear_sum_assignment(sentinel_cost)

    for (row, col) in zip(row_ind, col_ind):
        # If original was inf, then match only exists as solver was forced to pair up every row/col
        if np.isinf(cost[row, col]):
            unmatched_tracks.append(row)
            unmatched_detections.append(col)
        else:
            matches.append((row, col))

    # To address when M != N (pairs which would be left out above)
    M, N = cost.shape

    leftover_tracks = set(range(M)) - set(row_ind)
    leftover_detections = set(range(N)) - set(col_ind)

    unmatched_tracks.extend(leftover_tracks)
    unmatched_detections.extend(leftover_detections)

    return (matches, unmatched_tracks, unmatched_detections)
