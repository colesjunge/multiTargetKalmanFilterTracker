import numpy as np
from tracking.sim.sensors import PositionSensor



def test_clutter_count_is_poisson():
    """
    Run observe with no real targets (just clutter) and check that mean len of result is close to clutter_rate
    Use Poission var for error band
    """
    clutter_rate = 3.0
    n_calls = 300

    position_sensor = PositionSensor(sigma_pos=4.0, p_detect=1.0, clutter_rate=clutter_rate)
    rng = np.random.default_rng(42)

    results = []
    dt = .1

    for i in range(n_calls):
        results.append(len(position_sensor.observe(true_states=np.zeros((0,4)), t=i*dt, rng=rng)))

    assert abs(np.mean(results) - clutter_rate) < 4*(clutter_rate**.5)/(n_calls**.5), "Clutter is not following Poisson distribution (mean is not within 4 std clutter_rate accounting for n_calls)"


def test_seeded_rng_is_reproducible():
    """
    Ensure that the same seed produces the same clutter, detections, etc.
    """

    position_sensor = PositionSensor(sigma_pos=4.0, p_detect=0.5, clutter_rate=3.0)

    rng1 = np.random.default_rng(42) 
    rng2 = np.random.default_rng(42) # Same seed different object

    true_states = np.array([
        [1, 1, 1, 1],
        [1, 2, 2, 1],
        [4, 5, 5, 4],
        [3, 2, 1, 5]
        ])

    t = 3.0

    result1 = position_sensor.observe(true_states=true_states, t=t, rng=rng1)
    result2 = position_sensor.observe(true_states=true_states, t=t, rng=rng2)

    assert len(result1) == len(result2), "Results differ in length"

    for i in range(len(result1)):
        arr1 = result1[i].z
        arr2 = result2[i].z
        np.testing.assert_array_equal(arr1, arr2, "Position arrays not equal")

        assert result1[i].sensor_id == result2[i].sensor_id, "Sensor IDs not equal"

# pytest tests/test_sim.py -v 
