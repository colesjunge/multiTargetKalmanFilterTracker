import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml
config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

import numpy as np
from tracking.models.motion import ConstantVelocity2D
from tracking.models.measurement import LinearPosition2D
from tracking.sim.sensors import PositionSensor
from tracking.eval.metrics import position_rmse, match_tracks_to_truth, count_id_switches
from tracking.eval.viz import plot_sweep_curve
from tracking.tracker.manager import MultiTargetTracker
from tracking.sim.scenario import Scenario, build_scenario

# Set up variables from config
dt = config["simulation"]["dt"]
fov = tuple(config["simulation"]["fov"])
sigma_a = config["motion"]["sigma_a"] # Not passed through scenario
sigma_pos = config["measurement"]["sigma_pos"]
p_detect = config["sensor"]["p_detect"]
clutter_rate = config["sensor"]["clutter_rate"]
n_confirm = config["tracker"]["n_confirm"] # Not passed through scenario
max_misses = config["tracker"]["max_misses"] # Not passed through scenario
gate_confidence = config["tracker"]["gate_confidence"] # Not passed through scenario


def run_trial(scenario: Scenario, gate_confidence: float, match_gate: float, rng: np.random.Generator) -> tuple[float, int, int]:
    """
    Runs one full multi-target tracking trial for a given Scenario and scores the result against ground truth.


    In:  scenario; Scenario holding ground-truth trajectories and this trial's sensor/model parameters
         gate_confidence; chi-squared association gate confidence passed to MultiTargetTracker
         match_gate; radius (meters) for matching confirmed tracks to truth (evaluation only, never seen by the tracker)
         rng; random number generator, consumed by the sensor for noise/clutter/misses throughout the run
    Out: (mean_rmse, id_switches, never_matched_count)
         mean_rmse; position_rmse averaged across targets matched at least once this trial
         id_switches; total count_id_switches across every target in this trial
         never_matched_count; number of targets never matched to any confirmed track the entire run
    """


    # Model setup
    motion_model = ConstantVelocity2D(sigma_a) # No process noise
    measurement_model = LinearPosition2D(scenario.sigma_pos) # Measurement noise

    # Position sensor setup
    position_sensor = PositionSensor(scenario.sigma_pos, scenario.p_detect, scenario.clutter_rate, scenario.fov)

    # Multi Target Tracker setup
    mtt = MultiTargetTracker(motion=motion_model, measurement=measurement_model, n_confirm=n_confirm, max_misses=max_misses, gate_confidence=gate_confidence)

    assignment_log = []
    filtered_positions = {target_id: [] for target_id in scenario.trajectories}
    matched_true_positions = {target_id: [] for target_id in scenario.trajectories}

    for i in range(scenario.n_steps):
        t = i*scenario.dt

        true_states = []

        for target_id in scenario.trajectories.keys():
            true_states.append(scenario.trajectories[target_id][i])

        true_states = np.array(true_states)

        det = position_sensor.observe(true_states, t=t, rng=rng)

        result = mtt.step(det, scenario.dt, t)

        true_positions = {target_id: traj[i, :2] for target_id, traj in scenario.trajectories.items()}

        matched = match_tracks_to_truth(result.snapshots, true_positions, match_gate)

        for truth_id, track_id in matched.items():
            filtered_positions[truth_id].append(mtt.tracks[track_id].filter.x[:2])
            matched_true_positions[truth_id].append(true_positions[truth_id])

        assignment_log.append(matched)

    rmse_list = []
    never_matched_count = 0

    for target_id in scenario.trajectories.keys():
        if len(filtered_positions[target_id]) == 0:
            never_matched_count += 1
        else:
            rmse_list.append(position_rmse(np.array(filtered_positions[target_id]), np.array(matched_true_positions[target_id])))

    mean_rmse = np.mean(rmse_list)

    id_switches = count_id_switches(assignment_log, list(scenario.trajectories.keys()))

    return (float(mean_rmse), id_switches, never_matched_count)

# Main sweep
rng = np.random.default_rng(42) # Master RNG
n_targets = 10
spawn_region = tuple([0, 250, 0, 250])
n_steps = 100
match_gate = 20

clutter_rate_values = np.linspace(0, 10, num=10)
n_trials = 50

clutter_rate_stats = []

for sub_clutter_rate in clutter_rate_values:
    child_rngs = rng.spawn(n_trials)

    trial_rmses = []
    trial_id_switches = []
    trial_never_matched = []

    for child_rng in child_rngs:
        scenario = build_scenario(n_targets=n_targets,
                                n_steps=n_steps, 
                                dt=dt,
                                sigma_pos=sigma_pos,
                                p_detect=p_detect,
                                clutter_rate=sub_clutter_rate, 
                                fov=fov,
                                spawn_region=spawn_region,
                                rng=child_rng)

        mean_rmse, id_switches, never_matched = run_trial(scenario=scenario, gate_confidence=gate_confidence, match_gate=match_gate, rng=child_rng)

        trial_rmses.append(mean_rmse)
        trial_id_switches.append(id_switches)
        trial_never_matched.append(never_matched)
        

    agg_rmse = np.nanmean(trial_rmses)
    agg_id_switches = np.mean(trial_id_switches)
    agg_never_matched = np.mean(trial_never_matched)
    print(f"clutter_rate={sub_clutter_rate:.2f}: rmse={agg_rmse:.2f}, id_switches={agg_id_switches:.2f}, never_matched={agg_never_matched:.2f}")


    clutter_rate_stats.append((sub_clutter_rate, agg_rmse, agg_id_switches, agg_never_matched))

# Plot results
clutter_rate_values, agg_rmse_list, agg_id_switches_list, agg_never_matched_list = zip(*clutter_rate_stats)

clutter_rate_values = np.array(clutter_rate_values)
agg_rmse_list = np.array(agg_rmse_list)
agg_id_switches_list = np.array(agg_id_switches_list)
agg_never_matched_list = np.array(agg_never_matched_list)

# Clutter-rate vs RMSE
plot_sweep_curve(x_values=clutter_rate_values, y_values=agg_rmse_list, xlabel="Clutter Rates", ylabel="RMSE Values", title="Clutter vs RMSE", save_path="figures/run_sweep_rmse_plot.png")

# Clutter-rate vs ID Switches
plot_sweep_curve(x_values=clutter_rate_values, y_values=agg_id_switches_list, xlabel="Clutter Rates", ylabel="ID Switches", title="Clutter vs ID Switches", save_path="figures/run_sweep_id_switch_plot.png")

# Clutter-rate vs Never Matched
plot_sweep_curve(x_values=clutter_rate_values, y_values=agg_never_matched_list, xlabel="Clutter Rates", ylabel="Never Matched", title="Clutter vs Never Matched", save_path="figures/run_sweep_never_matched_plot.png")



