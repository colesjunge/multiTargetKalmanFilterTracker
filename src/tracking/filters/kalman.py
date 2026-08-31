from .base import BayesFilter
from ..models.motion import MotionModel
from ..models.measurement import MeasurementModel
import numpy as np
from scipy import linalg
from .utils import residual_with_angles

class KalmanFilter(BayesFilter):
    """
    Kalman Filter implementation for linear Gaussian systems.
    
    Predicts the state forward in time using a motion model
    Updates the state with a new measurement using a measurement model.
    Innovation represents the difference between predicted and actual measurements (used to compute the Kalman gain for the update step)
    """
    def __init__(self,
                 motion: MotionModel,
                 measurement: MeasurementModel,
                 x0: np.ndarray,    # (dim_x,) initial state
                 P0: np.ndarray):   # (dim_x, dim_x) initial covariance

        self.motion = motion
        self.measurement = measurement
        self.x = x0
        self.P = P0

    def predict(self, dt: float) -> None:
        """ Predict the state forward in time by dt seconds.
        In: dt; seconds since last update
        Mutates self.x, self.P in place. Returns nothing.
        """

        transition = self.motion.F(self.x, dt) # Calculated prior to updating self.x
        noise_cov = self.motion.Q(dt) # Same as above

        self.x = self.motion.f(self.x, dt)
        self.P = transition @ self.P @ transition.T + noise_cov

    def update(self, z: np.ndarray) -> None:
        """ Update the state with a new measurement z.
        In: z shape (dim_z,). 
        Mutates self.x, self.P in place. Returns nothing.
        """

        innovation_res, innovation_cov = self.innovation(z)
        selection_matrix = self.measurement.H(self.x)
        kalman_gain = linalg.solve(innovation_cov, selection_matrix @ self.P).T

        self.x = self.x + kalman_gain @ innovation_res
        self.P = (np.eye(self.motion.dim_x) - kalman_gain @ selection_matrix) @ self.P
        self.P = (self.P + self.P.T) / 2  # Corrects roundoff asymmetry
    def innovation(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Used by the association layer before committing to an update.
        In:  z shape (dim_z,); current measurement
        Out: (y, S) where y is (dim_z,) and S is (dim_z, dim_z); y is the innovation (residual) and S is the innovation covariance
        """

        innovation_res = residual_with_angles(z, self.measurement.h(self.x), self.measurement.angle_indices)
        selection_matrix = self.measurement.H(self.x)
        innovation_cov = selection_matrix @ self.P @ selection_matrix.T + self.measurement.R()

        return (innovation_res, innovation_cov)
