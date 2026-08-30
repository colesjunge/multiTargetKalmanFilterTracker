import matplotlib.pyplot as plt
import numpy as np
import pathlib

def plot_tracks(raw_measurements: list[np.ndarray], 
                filtered_est: list[np.ndarray], 
                true_states: list[np.ndarray], 
                save_path: str | None = None,
                turn_onset: np.ndarray | None = None) -> None:
    """
    Plots the tracks of targets over time.

    In: raw_measurements: list of np.ndarray, each of shape (T, 2) representing the raw measurements of a target over time.
        filtered_est: list of np.ndarray, each of shape (T, 2) representing the filtered estimates of a target over time.
        true_states: list of np.ndarray, each of shape (T, 2) representing the true states of a target over time.
        save_path: optional file path to save the figure to; if None, the plot is only shown, not saved.
        turn_onset: optional (2,) array [x, y] marking a specific point on the trajectory (where a maneuver begins) with a distinct marker.
    Out: None

    """

    plt.figure(figsize=(10, 8))

    # Plot raw measurements if provided
    if raw_measurements:
        for i, measurement in enumerate(raw_measurements):
            plt.plot(measurement[:, 0], measurement[:, 1], label=f'Raw Measurement {i+1}', linestyle='-', color='blue', marker='o', markevery=[0, -1])

    # Plot true states if provided
    if true_states:
        for i, true_state in enumerate(true_states):
            plt.plot(true_state[:, 0], true_state[:, 1], label=f'True State {i+1}', linestyle='--', color='gray', marker='s', markevery=[0, -1])

    # Plot filtered estimates if provided
    if filtered_est:
        for i, est in enumerate(filtered_est):
            plt.plot(est[:, 0], est[:, 1], label=f'Filtered Estimate {i+1}', linestyle=':', color='red', marker='^', markevery=[0, -1])

    plt.title('Target Tracks')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid()
    plt.axis('equal')

    if turn_onset is not None:
        plt.scatter(x=turn_onset[0], y=turn_onset[1], marker="*", label="Turn Onset", zorder=5)

    plt.legend()
    
    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.show()


def plot_sweep_curve(x_values: np.ndarray, 
                     y_values: np.ndarray, 
                     xlabel: str, 
                     ylabel: str, 
                     title: str, 
                     save_path: str | None = None) -> None:
    """
    Plots a single metric as a function of one swept scenario parameter as a line-and-marker curve.

    In: x_values: shape (N,), the swept parameter values (e.g. clutter_rate values tested).
        y_values: shape (N,), the corresponding aggregated metric value for each x_value (e.g. mean RMSE across trials).
        xlabel: label for the x-axis (name of the swept parameter).
        ylabel: label for the y-axis (name of the metric being plotted).
        title: plot title.
        save_path: optional file path to save the figure to; if None, the plot is only shown, not saved.
    Out: None
    """

    plt.figure(figsize=(10, 8))
    plt.plot(x_values, y_values, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()

    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.show()
