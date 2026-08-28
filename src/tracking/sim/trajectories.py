import numpy as np

def constant_velocity_trajectory(x0: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
    """Propagates the state forward with the exact constant velocity physics with zero process noise
    This is ground truth, not a filter estimate
    
    In:  x0 shape (4,), dt, n_steps; Initial [px, py, vx, vy], seconds; step count
    Out: shape (n_steps, 4); True state at each of n_steps time steps
    """

    transition = np.array([
                        [1, 0, dt, 0], 
                        [0, 1, 0, dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                               ])

    trajectory = np.zeros((n_steps, 4))
    trajectory[0] = x0

    for i in range(1, n_steps):
        trajectory[i] = transition @ trajectory[i-1]

    return trajectory

def coordinated_turn_trajectory(x0: np.ndarray, dt: float, n_steps: int, omega: float) -> np.ndarray:
    """ Includes turn under constant rate and speed
    In: x0 shape (4,), dt, n_steps, omega; Initial [px, py, vx, vy], seconds; step count; turn rate in rad/s (pos is CCW)
    Out: shape (n_steps, 4); True state at each of n_steps time steps  
    """
    velocity_update = np.array([
                        [np.cos(omega*dt), -np.sin(omega*dt)],
                        [np.sin(omega*dt), np.cos(omega*dt)]
                        ])

    position_update = np.array([
                        [np.sin(omega*dt), np.cos(omega*dt) - 1],
                        [1 - np.cos(omega*dt), np.sin(omega*dt)]
                        ])

    transition = np.block([
                        [np.eye(2), (1/omega)* position_update],
                        [np.zeros((2,2)), velocity_update]
                        ])
    
    trajectory = np.zeros((n_steps, 4))
    trajectory[0] = x0

    for i in range(1, n_steps):
        trajectory[i] = transition @ trajectory[i-1]

    return trajectory