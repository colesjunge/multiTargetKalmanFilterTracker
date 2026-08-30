import numpy as np
from dataclasses import dataclass

from tracking.sim.trajectories import constant_velocity_trajectory

@dataclass
class Scenario():
    """
    Scenario dataclass that includes metrics and target trajectories
    """

    trajectories: dict[int, np.ndarray] # target_id -> ground_truth (n_steps, 4)
    dt: float
    n_steps: int
    sigma_pos: float
    p_detect: float
    clutter_rate: float
    fov: tuple


def build_scenario(n_targets: int, 
                   n_steps: int, 
                   dt: float, 
                   sigma_pos: float, 
                   p_detect: float, 
                   clutter_rate: float, 
                   fov: tuple,
                   spawn_region: tuple, # Distinct from fov, restrict spawn region so targets cross paths
                   rng: np.random.Generator) -> Scenario:

    # generate trajectories
    trajectories = {}
    for target_id in range(n_targets):

        # Determine initial position
        init_x = rng.uniform(spawn_region[0], spawn_region[1])
        init_y = rng.uniform(spawn_region[2], spawn_region[3])

        # Determine initial velocity
        speed = rng.uniform(10, 30) # Small drones go about 10-30 m/s
        theta = rng.uniform(0, 2*np.pi)
        init_vx = speed*np.cos(theta)
        init_vy = speed*np.sin(theta)

        x0 = np.array([init_x, init_y, init_vx, init_vy])

        trajectories[target_id] = constant_velocity_trajectory(x0, dt, n_steps)


    return Scenario(trajectories=trajectories, 
                    dt=dt,
                    n_steps=n_steps,
                    sigma_pos=sigma_pos,
                    p_detect=p_detect,
                    clutter_rate=clutter_rate,
                    fov=fov)
