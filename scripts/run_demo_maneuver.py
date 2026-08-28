import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from tracking.sim.trajectories import constant_velocity_trajectory, coordinated_turn_trajectory
from tracking.models.motion import ConstantVelocity2D
from tracking.models.measurement import LinearPosition2D
from tracking.sim.sensors import PositionSensor
from tracking.tracker.manager import MultiTargetTracker
from tracking.eval.metrics import position_rmse, peak_position_error
from tracking.eval.viz import plot_tracks

rng = np.random.default_rng(42)  # For reproducibility

dt = 0.1
n_steps_straight = 51
n_steps_turn = 50

# Set up target (first leg is striaght, second has turn)
x0 = np.array([0.0, 0.0, 2.0, 2.0])
straight_trajectory = constant_velocity_trajectory(x0, dt, n_steps_straight)
turn_trajectory = coordinated_turn_trajectory(x0=straight_trajectory[-1], dt=dt, n_steps=n_steps_turn, omega=.785) # ~225 deg / 3.93 rad turn

trajectory = np.concatenate((straight_trajectory[:-1], turn_trajectory))

# This is to deal with repeated state at end of straight and beginning of turn
n_steps_straight -= 1
n_steps = n_steps_straight + n_steps_turn


# Model setup
motion_model = ConstantVelocity2D(sigma_a=0.0) # No process noise
measurement_model = LinearPosition2D(sigma_pos=4.0) # Measurement noise 
position_sensor = PositionSensor(sigma_pos=4.0, p_detect=1.0, clutter_rate=0.0) # No clutter for this demo
mtt = MultiTargetTracker(motion=motion_model, measurement=measurement_model, gate_confidence=.999) # Higher gate tolerance so lagging track can be kept alive (part of this demo)

# Main loop (Keeping clutter mechanics from here forward, regardless of if clutter is used)
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

        # If unmatched, it just spawned this frame and new track can be found by the seeded state
        if real_track_id is None:
            real_track_id = next((track.track_id for track in mtt.tracks.values() if np.array_equal(track.filter.x[:2], det[real_idx].z)), None)

        if real_track_id is not None:
            raw.append(det[real_idx].z)
            filtered.append(mtt.tracks[real_track_id].filter.x[:2])
            true.append(trajectory[i][:2])

filtered_arr = np.array(filtered)
raw_arr = np.array(raw)
true_arr = np.array(true)


# Calculate RMSE between ground truth trajectories and filtered estimates / raw measurements
# One for overall, one for straight leg, one for turn leg
total_filtered_rmse = position_rmse(filtered_arr, true_arr)
print(f"Root Mean Square Error (Filtered Estimates) for Total Track: {total_filtered_rmse}")
total_raw_rmse = position_rmse(raw_arr, true_arr)
print(f"Root Mean Square Error (Raw Measurements) for Total Track: {total_raw_rmse}")

straight_filtered_rmse = position_rmse(filtered_arr[:n_steps_straight], true_arr[:n_steps_straight])
print(f"Root Mean Square Error (Filtered Estimates) for Straight Leg: {straight_filtered_rmse}")
straight_raw_rmse = position_rmse(raw_arr[:n_steps_straight], true_arr[:n_steps_straight])
print(f"Root Mean Square Error (Raw Measurements) for Straight Leg: {straight_raw_rmse}")

turn_filtered_rmse = position_rmse(filtered_arr[-n_steps_turn:], true_arr[-n_steps_turn:])
print(f"Root Mean Square Error (Filtered Estimates) for Turn Leg: {turn_filtered_rmse}")
turn_raw_rmse = position_rmse(raw_arr[-n_steps_turn:], true_arr[-n_steps_turn:])
print(f"Root Mean Square Error (Raw Measurements) for Turn Leg: {turn_raw_rmse}")

# Calculate peak error for each frame (for each leg)
peak_error_straight, peak_error_frame_straight = peak_position_error(filtered_arr[:n_steps_straight], true_arr[:n_steps_straight])
print(f"Straight Leg Peak error {peak_error_straight} on frame {peak_error_frame_straight}")

peak_error_turn, peak_error_frame_turn = peak_position_error(filtered_arr[-n_steps_turn:], true_arr[-n_steps_turn:])
print(f"Turn Leg Peak error {peak_error_turn} on frame {peak_error_frame_turn + len(filtered_arr) - n_steps_turn}")

# Plot Results
plot_tracks([raw_arr], [filtered_arr], [true_arr], save_path="figures/run_demo_maneuver_plot.png", turn_onset=trajectory[n_steps_straight][:2])