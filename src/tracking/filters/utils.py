# These will be used to help implement radar measurements with states [range, bearing]

import numpy as np




def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    """ Wraps angle(s) to (-pi, pi]
    In: a; array of angle(s)
    Out: wrapped array of angle(s)
    """

    return np.arctan2(np.sin(a), np.cos(a))


def residual_with_angles(a: np.ndarray, b: np.ndarray, angle_idx: list[int]) -> np.ndarray:
    """Difference between angles using wrap to ensure correct residual
    In: a; shape (dim_z, ), b; shape (dim_z, ), angle_idx; what indices refer to angles
    Out: diff, residuals with angle diff wrapped
    """

    diff = a - b

    for i in angle_idx:
        diff[i] = wrap_angle(diff[i])

    return diff




