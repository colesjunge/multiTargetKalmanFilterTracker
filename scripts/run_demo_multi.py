import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from tracking.sim.trajectories import constant_velocity_trajectory
from tracking.models.motion import ConstantVelocity2D
from tracking.models.measurement import LinearPosition2D
from tracking.sim.sensors import PositionSensor
from tracking.eval.metrics import position_rmse
from tracking.eval.viz import plot_tracks
from tracking.tracker.manager import MultiTargetTracker

rng = np.random.default_rng(42)  # For reproducibility

dt = 0.1
n_steps = 100

# Multiple independent true trajectories

x0_true_0 = np.array([0, 0, 2, 2]) # px, py, vx, vy
x0_true_1 = np.array([0, 20, 2, -2]) # px, py, vx, vy
x0_true_2 = np.array([25, 0, -1, 2]) # px, py, vx, vy
x0_true_3 = np.array([25, 25, -2, -1]) # px, py, vx, vy
x0_true_4 = np.array([12, 5, 0.2, 3]) # px, py, vx, vy

# Ground truth trajectories
trajectory_0 = constant_velocity_trajectory(x0_true_0, dt, n_steps)
trajectory_1 = constant_velocity_trajectory(x0_true_1, dt, n_steps)
trajectory_2 = constant_velocity_trajectory(x0_true_2, dt, n_steps)
trajectory_3 = constant_velocity_trajectory(x0_true_3, dt, n_steps)
trajectory_4 = constant_velocity_trajectory(x0_true_4, dt, n_steps)

trajectories = [trajectory_0, trajectory_1, trajectory_2, trajectory_3, trajectory_4]

# Model setup
motion_model = ConstantVelocity2D(sigma_a=0.0) # No process noise
measurement_model = LinearPosition2D(sigma_pos=4.0) # Measurement noise

# Position sensor setup
position_sensor = PositionSensor(sigma_pos=4.0, p_detect=1.0, clutter_rate=0.0)

# Run the simulation
filtered_by_target = [[],[],[],[],[]]
raw_by_target = [[],[],[],[],[]]


mtt = MultiTargetTracker(motion=motion_model, measurement=measurement_model)
track_ids = []

#Initial step
for k in range(5):
    first_det = position_sensor.observe(trajectories[k][0:1], t=0.0, rng=rng)[0]
    track = mtt._init_track(first_det, timestamp=0.0)
    track_ids.append(track.track_id)

# Rest of simulation
for i in range(1, n_steps):
    dets_by_track_id = {}
    for k in range(5):
        measurements = position_sensor.observe(trajectories[k][i:i+1], t=i*dt, rng=rng)
        if measurements:
            dets_by_track_id[track_ids[k]] = measurements[0]
            raw_by_target[k].append(measurements[0].z)
    result = mtt.step_cheated(dets_by_track_id, dt, i*dt)
    for k in range(5):
        filtered_by_target[k].append(mtt.tracks[track_ids[k]].filter.x[:2])


filtered_arrs = [np.array(lst) for lst in filtered_by_target]
raw_arrs = [np.array(lst) for lst in raw_by_target]
true_arrs = [traj[1:n_steps, :2] for traj in trajectories]

# Calculate RMSE between ground truth trajectories and filtered estimates / raw measurements
for k in range(5):
    filtered_rmse = position_rmse(filtered_arrs[k], true_arrs[k])
    print(f"Root Mean Square Error (Filtered Estimates) for Target {k}: {filtered_rmse}")
    raw_rmse = position_rmse(raw_arrs[k], true_arrs[k])
    print(f"Root Mean Square Error (Raw Measurements) for Target {k}: {raw_rmse}")

# Plot Results
plot_tracks(raw_arrs, filtered_arrs, true_arrs, save_path="figures/run_demo_multi_plot.png")
