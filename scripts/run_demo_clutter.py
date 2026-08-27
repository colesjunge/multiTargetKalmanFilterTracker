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

# Set up target
x0 = np.array([0.0, 0.0, 2.0, 2.0])
trajectory = constant_velocity_trajectory(x0, dt, n_steps)

# Model setup
motion_model = ConstantVelocity2D(sigma_a=0.0) # No process noise
measurement_model = LinearPosition2D(sigma_pos=4.0) # Measurement noise
position_sensor = PositionSensor(sigma_pos=4.0, p_detect=0.8, clutter_rate=1.0)
mtt = MultiTargetTracker(motion=motion_model, measurement=measurement_model)

# Main loop
filtered = []
raw = []
true =[]

confirmed_track_ids = set()

for i in range(0, n_steps):
    t = i*dt
    det = position_sensor.observe(trajectory[i:i+1], t=t, rng=rng)

    real_idx = next((i for i, d in enumerate(det) if d.sensor_id == "radar"), None)

    result = mtt.step(det, dt, t)

    for snapshot in result.snapshots:
        if snapshot.confirmed:
            confirmed_track_ids.add(snapshot.track_id)

    if real_idx is not None:
        real_track_id = next((k for k, v in result.assignments.items() if v == real_idx), None)

        # If unmatched, it just spawned this frame and new track can be foudn by the seeded state
        if real_track_id is None:
            real_track_id = next((track.track_id for track in mtt.tracks.values() if np.array_equal(track.filter.x[:2], det[real_idx].z)), None)

        if real_track_id is not None:
            raw.append(det[real_idx].z)
            filtered.append(mtt.tracks[real_track_id].filter.x[:2])
            true.append(trajectory[i][:2])

filtered_arr = np.array(filtered)
raw_arr = np.array(raw)
true_arr = np.array(true)

# Confirm that real_track_id still in tracks
print(f"Real track ID in tracks: {real_track_id in mtt.tracks}")

# Check whether there is only one confirmed track and id matches the real_track_id
print(f"Confirmed track amount: {len(confirmed_track_ids)}")
print(f"Confirmed track ID: {confirmed_track_ids}")
print(f"Real track ID: {real_track_id}")

# Calculate RMSE between ground truth trajectories and filtered estimates / raw measurements
filtered_rmse = position_rmse(filtered_arr, true_arr)
print(f"Root Mean Square Error (Filtered Estimates) for Target A: {filtered_rmse}")
raw_rmse = position_rmse(raw_arr, true_arr)
print(f"Root Mean Square Error (Raw Measurements) for Target A: {raw_rmse}")

# Plot Results
plot_tracks([raw_arr], [filtered_arr], [true_arr], save_path="figures/run_demo_clutter_plot.png")



