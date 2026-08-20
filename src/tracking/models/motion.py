from abc import ABC, abstractmethod
import numpy as np

class MotionModel(ABC):
    """
    Abstract base class for motion models

    Supplies the state transition and process noise for the predict step
    Future subclasses will implement nonlinear and learned variants (why they are functions and not just matrices)
    """
    dim_x: int

    @abstractmethod
    def f(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Propagate state forward.
        In:  x  shape (dim_x,), dt; current state estimate, seconds since last update
        Out: shape (dim_x,); predicted state
        """

    @abstractmethod
    def F(self, x: np.ndarray, dt: float) -> np.ndarray:
        """State transition Jacobian (this is the constant matrix F in the linear case).
        In:  x shape (dim_x,), dt; current state estimate, seconds since last update
        Out: shape (dim_x, dim_x); Jacobian of f w.r.t. x, evaluated at x
        """


    @abstractmethod
    def Q(self, dt: float) -> np.ndarray:
        """Process noise covariance.
        In:  dt; seconds since last update
        Out: shape (dim_x, dim_x); symmetric positive semi-definite matrix (uncertainty due to unmoddeled acceleration)
        """

class ConstantVelocity2D(MotionModel):
    """
    Constant velocity motion model in 2D (x, y)
    State = [px, py, vx, vy]
    This is the standard linear motion model used in Kalman filters.
    """

    dim_x = 4

    def __init__(self, sigma_a: float):
        self.sigma_a = sigma_a # std-dev of unmodeled acceleration (m/s^2)

    def f(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Propagate state forward.
        In:  x  shape (4,), dt; current state estimate, seconds since last update
        Out: shape (4,); predicted state
        """

        transition = self.F(x, dt)
        pred_x = transition @ x

        return pred_x

    def F(self, x: np.ndarray, dt: float) -> np.ndarray:
        """State transition Jacobian.
        In:  x shape (4,), dt; current state estimate, seconds since last update
        Out: shape (4, 4); Jacobian of f w.r.t. x, evaluated at x (constant in this case)
        """

        transition = np.array([
                    [1, 0, dt, 0], 
                    [0, 1, 0, dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                           ])
        
        return transition
    
    def Q(self, dt: float) -> np.ndarray:
        """Process noise covariance.
        In:  dt; seconds since last update
        Out: shape (4, 4); symmetric positive semi-definite matrix (uncertainty due to unmoddeled acceleration)
        """
    
        noise_cov = np.array([
                    [dt**4/4, 0, dt**3/2, 0],
                    [0, dt**4/4, 0, dt**3/2],
                    [dt**3/2, 0, dt**2, 0],
                    [0, dt**3/2, 0, dt**2]
                  ]) * self.sigma_a**2

        return noise_cov
