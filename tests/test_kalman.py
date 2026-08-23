import numpy as np
from tracking.models.motion import ConstantVelocity2D
from tracking.models.measurement import LinearPosition2D
from tracking.filters.kalman import KalmanFilter
from tracking.sim.trajectories import constant_velocity_trajectory

def test_converges_to_truth_with_exact_measurements():
    """
    Check CV filter w/ zero noise
    Goal is position error -> ~0 and P's trace shrinks and stabilizes over a straight-line trajectory.
    """

    dt = 0.1
    n_steps = 100
    x0_true = np.array([0, 0, 1, 1]) # px, py, vx, vy

    # Ground truth trajectory
    trajectory = constant_velocity_trajectory(x0_true, dt, n_steps)

    # Filter setup
    motion_model = ConstantVelocity2D(sigma_a=0.0) # No process noise
    measurement_model = LinearPosition2D(sigma_pos=1e-6) # Small measurement noise to avoid errors in the Kalman gain calculation

    x0_filter = np.array([.5, .3, 5, 10]) # Initial state uncertainty (random numbers I've chosen)
    P0 = np.array([[10, 0, 0, 0], 
                   [0, 12, 0, 0], 
                   [0, 0, 34, 0], 
                   [0, 0, 0, 35]]) # Initial uncertainty (random numbers I've chosen)

    kf = KalmanFilter(motion_model, measurement_model, x0=x0_filter, P0=P0)
    traces = []

    for i in range(1, n_steps):
        kf.predict(dt)
        kf.update(trajectory[i][:2]) # Update with exact position measurement
        traces.append(np.trace(kf.P))

        if i > 2:  # Skip first few steps to allow filter to converge
            # Check that the filter's state is close to the ground truth
            assert np.allclose(kf.x[:2], trajectory[i][:2], atol=1e-6), f"Position mismatch at step {i}"
            assert np.allclose(kf.x[2:], trajectory[i][2:], atol=1e-6), f"Velocity mismatch at step {i}"

            # Check that the covariance is shrinking and stabilizing
            assert traces[-2] - traces[-1] > -1e-6, f"Covariance trace too large at step {i}"

# To run: pytest tests/test_kalman.py -v