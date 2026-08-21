import numpy as np
from tracking.eval.metrics import nees, average_nees
from tracking.sim.trajectories import constant_velocity_trajectory
from tracking.models.motion import ConstantVelocity2D
from tracking.models.measurement import LinearPosition2D
from tracking.filters.kalman import KalmanFilter
from tracking.sim.sensors import PositionSensor



def test_nees_of_perfect_estimate_is_zero():
    """
    Check NEES with perfect estimate is zero
    """

    P = np.array([[3, 0], 
                [0, 3]])
    x = np.array([5, 6])

    assert nees(x, P, x) == 0, "NEES not 0 when x_est == x_true"


def test_nees_averages_to_state_dim_for_consistent_filter():
    """
    100 independent noisy runs of the same true trajectory;
    average NEES over the whole simulation should land near dim_x (4).
    """

    rng = np.random.default_rng(42)  # For reproducibility
    rngs = rng.spawn(100)

    dt = 0.1
    n_steps = 100
    x0_true = np.array([0, 0, 1, 1]) # px, py, vx, vy

    # Ground truth trajectory
    trajectory = constant_velocity_trajectory(x0_true, dt, n_steps)

    # Position sensor setup
    position_sensor = PositionSensor(sigma_pos=4.0, p_detect=1.0, clutter_rate=0.0)

    # Run the simulation
    all_snapshots = []
    all_truths = []

    # 100 Path MC Simulation
    for i in range(100):
        curr_rng = rngs[i]

        x0_filter = np.array([.5, .3, 5, 10]) # Initial state uncertainty (random numbers I've chosen)
        P0 = np.array([[10, 0, 0, 0], 
                        [0, 12, 0, 0], 
                        [0, 0, 34, 0], 
                        [0, 0, 0, 35]]) # Initial uncertainty (random numbers I've chosen)
    
        # Filter setup
        motion_model = ConstantVelocity2D(sigma_a=0.0) # No process noise
        measurement_model = LinearPosition2D(sigma_pos=4.0) # Measurement noise
        kf = KalmanFilter(motion_model, measurement_model, x0=x0_filter, P0=P0)

        for j in range(1, n_steps):
            all_truths.append(trajectory[j])

            measurements = position_sensor.observe(trajectory[j:j+1], t=j*dt, rng=curr_rng)

            kf.predict(dt)
            if measurements:
                kf.update(measurements[0].z)

            all_snapshots.append((kf.x.copy(), kf.P.copy()))

    # Call Average NEES and check that it is around 4
    avg_nees = average_nees(all_snapshots, all_truths)

    assert abs(avg_nees - 4) < 1, "Average NEES to far from 4 (dim_x)"
            


                

