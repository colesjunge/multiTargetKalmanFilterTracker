from abc import ABC, abstractmethod
import numpy as np

class MeasurementModel(ABC):
    """
    Abstract base class for measurement models

    Supplies the measurement function and measurement noise for the update step
    Future subclasses will implement nonlinear and learned variants (why they are functions and not just matrices)
    """
    dim_z: int

    @abstractmethod
    def h(self, x: np.ndarray) -> np.ndarray:
        """Predicted measurement given state.
        In:  x shape (dim_x,); current state estimate
        Out: shape (dim_z,); predicted measurement
        """

    @abstractmethod
    def H(self, x: np.ndarray) -> np.ndarray:
        """Measurement Jacobian.
        In:  x shape (dim_x,); current state estimate
        Out: shape (dim_z, dim_x); Jacobian of h w.r.t. x, evaluated at x
        """

    @abstractmethod
    def R(self) -> np.ndarray:
        """Measurement noise covariance. 
        Out: shape (dim_z, dim_z); symmetric positive semi-definite matrix (uncertainty due to sensor noise)
        """

class LinearPosition2D(MeasurementModel):
    """
    Linear Position Measurement Model
    Measurement = [px, py]
    No information on velocity, only position
    """
    dim_z = 2

    def __init__(self, sigma_pos: float): 
        self.sigma_pos = sigma_pos  # std-dev of position measurement noise (m)

    def h(self, x: np.ndarray) -> np.ndarray:
        """Predicted measurement given state.
        In:  x shape (4,); current state estimate
        Out: shape (2,); predicted measurement
        """
        selection_matrix = self.H(x)
        pred_z = selection_matrix @ x

        return pred_z

    def H(self, x: np.ndarray) -> np.ndarray:
        """Measurement Jacobian.
        In:  x shape (4,); current state estimate
        Out: shape (2, 4); Jacobian of h w.r.t. x, evaluated at x
        """

        selection_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        return selection_matrix
    
    def R(self) -> np.ndarray:
        """Measurement noise covariance. 
        Out: shape (2, 2); symmetric positive semi-definite matrix (uncertainty due to sensor noise)
        """
        noise_cov = np.array([
            [self.sigma_pos**2, 0],
            [0, self.sigma_pos**2]
        ])

        return noise_cov