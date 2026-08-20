from abc import ABC, abstractmethod
import numpy as np

class BayesFilter(ABC):
    """
    Interface every filter obeys
    
    Future subclasses will implement nonlinear and learned variants
    """
    x: np.ndarray   # (dim_x,)
    P: np.ndarray   # (dim_x, dim_x)

    @abstractmethod
    def predict(self, dt: float) -> None:
        """ Predict the state forward in time by dt seconds.
        In: dt; seconds since last update
        Mutates self.x, self.P in place. Returns nothing.
        """

    @abstractmethod
    def update(self, z: np.ndarray) -> None:
        """ Update the state with a new measurement z.
        In: z shape (dim_z,). 
        Mutates self.x, self.P in place. Returns nothing.
        """

    @abstractmethod
    def innovation(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Used by the association layer before committing to an update.
        In:  z shape (dim_z,); current measurement
        Out: (y, S) where y is (dim_z,) and S is (dim_z, dim_z); y is the innovation (residual) and S is the innovation covariance
        """
