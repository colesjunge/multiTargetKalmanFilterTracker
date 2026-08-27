from abc import ABC, abstractmethod
import numpy as np

from tracking.types import Detection


class Sensor(ABC):
    """
    Abstract base class for sensors

    Eventually will be used to simulate different types of sensors (IR, EO, etc.)
    """
    dim_z: int

    @abstractmethod
    def observe(self, true_states: np.ndarray, t: float, rng: np.random.Generator) -> list[Detection]:
        """Observe the true state of the world and return a list of detections.
        In:  true_states shape (n_targets, dim_x), t; current time, rng; random number generator
        Out: list of Detection objects (length will vary given number of objects)
        """

class PositionSensor(Sensor):
    """
    Position sensor

    Measures the position of targets in 2D space with Gaussian noise.
    """

    def __init__(self, sigma_pos: float, p_detect: float = 1.0, clutter_rate: float = 0.0, fov: tuple = (0,10000,0,10000)):
        """
        sigma_pos: std-dev of position measurement noise (m)
        p_detect: probability each true target is actually detected.
        clutter_rate: mean number of false alarms per frame (Poisson).
        fov: field of view (x_min, x_max, y_min, y_max)
        """

        self.sigma_pos = sigma_pos
        self.p_detect = p_detect
        self.clutter_rate = clutter_rate
        self.fov = fov
        self.dim_z = 2  # Position sensor measures (px, py)

    def observe(self, true_states: np.ndarray, t: float, rng: np.random.Generator) -> list[Detection]:
        """Adds noise to true positions and returns a list of Detection objects.
        In:  true_states shape (n_targets, dim_x), t; current time, rng; random number generator
        Out: list of Detection objects (length will vary given number of objects)
        """

        # Get the number of targets
        n_targets = true_states.shape[0]

        # Determine which targets are detected
        detected_mask = rng.uniform(size=n_targets) < self.p_detect

        # Generate position measurements for detected targets
        measurements = []
        for i in range(n_targets):
            true_pos = true_states[i][:2]  # Extract true position (px, py)

            if (self.fov[0] <= true_pos[0] <= self.fov[1]) and (self.fov[2] <= true_pos[1] <= self.fov[3]):
                if detected_mask[i]:
                    # Add Gaussian noise to the true position
                    noise = rng.normal(loc=0.0, scale=self.sigma_pos, size=2)
                    measurement = true_pos + noise

                    measurements.append(Detection(measurement, t, sensor_id="radar", meta={}))

        # Add clutter to measurements
        n_clutter = rng.poisson(self.clutter_rate) # Poisson as clutter_rate is defined as mean count per frame
        for _ in range(n_clutter):
            # Clutter X, Y (uniform from fov bounds)
            clutter_pos = np.array([
                rng.uniform(self.fov[0], self.fov[1]),
                rng.uniform(self.fov[2], self.fov[3])
            ])

            measurements.append(Detection(clutter_pos, t, sensor_id="clutter", meta={"clutter": True}))

        return measurements

        
        
