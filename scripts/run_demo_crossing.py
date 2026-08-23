import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from tracking.sim.trajectories import constant_velocity_trajectory
from tracking.models.motion import ConstantVelocity2D
from tracking.models.measurement import LinearPosition2D
from tracking.sim.sensors import PositionSensor
from tracking.tracker.manager import MultiTargetTracker
from tracking.eval.metrics import position_rmse
from tracking.eval.viz import plot_tracks

rng = np.random.default_rng(7)  # For reproducibility

dt = 0.1
n_steps = 100

# Set up two targets on crossing path (These will met up at (10,10) at t = 5.0)
x0_a = np.array([0.0, 0.0, 2.0, 2.0])
x0_b = np.array([0.0, 20.0, 2.0, -2.0])
trajectory_a = constant_velocity_trajectory(x0_a, dt, n_steps)
trajectory_b = constant_velocity_trajectory(x0_b, dt, n_steps)

# Model setup
motion_model = ConstantVelocity2D(sigma_a=0.0) # No process noise
measurement_model = LinearPosition2D(sigma_pos=4.0) # Measurement noise
position_sensor = PositionSensor(sigma_pos=4.0, p_detect=1.0, clutter_rate=0.0)
mtt = MultiTargetTracker(motion=motion_model, measurement=measurement_model)

# Initialize both tracks (initial id labeling, but after left up to step())
det_a0 = position_sensor.observe(trajectory_a[0:1], t=0.0, rng=rng)[0]
det_b0 = position_sensor.observe(trajectory_b[0:1], t=0.0, rng=rng)[0]
track_a = mtt._init_track(det_a0, timestamp=0.0)
track_b = mtt._init_track(det_b0, timestamp=0.0)
id_a, id_b = track_a.track_id, track_b.track_id


# Main loop
filtered_a, filtered_b = [],[]
raw_a, raw_b = [],[]

for i in range(1, n_steps):
    t = i*dt
    det_a = position_sensor.observe(trajectory_a[i:i+1], t=t, rng=rng)
    det_b =position_sensor.observe(trajectory_b[i:i+1], t=t, rng=rng)
    
    detections = det_a + det_b

    if det_a:
        raw_a.append(det_a[0].z)
    if det_b:
        raw_b.append(det_b[0].z)

    result = mtt.step(detections, dt, t)

    filtered_a.append(mtt.tracks[id_a].filter.x[:2])
    filtered_b.append(mtt.tracks[id_b].filter.x[:2])

filtered_arr_a = np.array(filtered_a)
filtered_arr_b = np.array(filtered_b)
raw_arr_a = np.array(raw_a)
raw_arr_b = np.array(raw_b)

true_arr_a = trajectory_a[1:n_steps, :2]
true_arr_b = trajectory_b[1:n_steps, :2]

# Calculate RMSE between ground truth trajectories and filtered estimates / raw measurements

filtered_rmse_a = position_rmse(filtered_arr_a, true_arr_a)
print(f"Root Mean Square Error (Filtered Estimates) for Target A: {filtered_rmse_a}")
raw_rmse_a = position_rmse(raw_arr_a, true_arr_a)
print(f"Root Mean Square Error (Raw Measurements) for Target A: {raw_rmse_a}")

filtered_rmse_b = position_rmse(filtered_arr_b, true_arr_b)
print(f"Root Mean Square Error (Filtered Estimates) for Target B: {filtered_rmse_b}")
raw_rmse_b = position_rmse(raw_arr_b, true_arr_b)
print(f"Root Mean Square Error (Raw Measurements) for Target B: {raw_rmse_b}")

# Plot Results
plot_tracks([raw_arr_a, raw_arr_b], [filtered_arr_a, filtered_arr_b], [true_arr_a, true_arr_b], save_path="figures/run_demo_crossing_plot.png")

