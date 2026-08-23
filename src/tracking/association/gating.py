import numpy as np
from scipy import linalg, stats

from ..tracker.track import Track
from ..types import Detection



def mahalanobis_sq(y: np.ndarray, S: np.ndarray) -> float:
    """ Calculate the Mahalanobis distance

    In: y shape (n,), S shape (n, n)
    Out: d_sq scalar
    """

    if S.shape[0] != S.shape[1] or S.shape[0] != y.shape[0]:
        raise ValueError("Innovation Covariance matrix S must be square and match the dimension of the state.")

    # Calculate distance
    x = linalg.solve(S, y)
    d_sq = y @ x

    return float(d_sq)


def chi2_gate_threshold(dim_z: int, confidence: float = 0.99) -> float:
    """ Calculate the the cutoff for the squared Mahalanobis distance
    from the chi squared distributions confidence percentile at dim_z degrees of freedom

    In: dim_z int, confidence float
    Out: scalar
    """


    gate_threshold = stats.chi2.ppf(confidence, df=dim_z)
    return float(gate_threshold)


def build_cost_matrix(tracks: list[Track], detections: list[Detection], gate: float) -> np.ndarray:
    """ Create cost matrix using track and detection pairs using mahalanobis distance and cutoffs

    In: tracks shape(M?), detections shape(N?), gate scalar
    Out: cost_matrix shape(M, N)
    """

    M = len(tracks)
    N = len(detections)
    cost = np.zeros((M, N), dtype=float)

    for i in range(M):
        for j in range(N):
            y, S = tracks[i].filter.innovation(detections[j].z)
            d_sq = mahalanobis_sq(y, S)
            cost[i, j] = d_sq if d_sq <= gate else np.inf

    return cost




