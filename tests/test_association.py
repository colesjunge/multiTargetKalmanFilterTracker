import numpy as np
from tracking.types import Detection
from tracking.models.motion import ConstantVelocity2D
from tracking.models.measurement import LinearPosition2D
from tracking.tracker.manager import MultiTargetTracker
from tracking.association.gating import chi2_gate_threshold, build_cost_matrix
from tracking.association.assignment import hungarian_assign


def test_gate_rejects_distant_detection():
    """
    Builds a track and detection far outside plausible range given by predicted state/covariance
    Assert the resulting cost matrix entry is inf
    """
      
    motion = ConstantVelocity2D(sigma_a=0.0)
    measurement = LinearPosition2D(sigma_pos=4.0)
    mtt = MultiTargetTracker(motion, measurement)

    # One track seeded near the origin
    track = mtt._init_track(Detection(z=np.array([0.0, 0.0]), timestamp=0.0), timestamp=0.0)
    track.predict(dt=0.1)

    # Detection is nowhere near where this track could plausibly be
    far_det = Detection(z=np.array([1000, 1000]), timestamp=0.1)

    gate = chi2_gate_threshold(dim_z=2, confidence=0.99)
    cost = build_cost_matrix([track], [far_det], gate)

    assert np.isinf(cost[0, 0]), "Gate did not reject implausible detection"

def test_hungarian_prefers_global_optimum():
    """
    Construct cost matrix by hand where picking cheapest option per row greedily gives worse total
    than global optimal pairing
    Assert hungarian_assign finds optimal solution, not greedy
    """

    cost_track0_det0 = 5
    cost_track0_det1 = 10
    cost_track1_det0 = 20
    cost_track1_det1 = 50

    cost_matrix = np.array([
        [cost_track0_det0, cost_track0_det1],
        [cost_track1_det0, cost_track1_det1],
])

    assert cost_track0_det1 + cost_track1_det0 < cost_track0_det0 + cost_track1_det1, "Hand picked matrix does not work for this context"

    (matches, unmatched_tracks, unmatched_detections) = hungarian_assign(cost_matrix)

    assert set(matches) == {(0, 1), (1, 0)}, "Correct globally optimal matches"
    assert unmatched_tracks == [], "Tracks have gone unmatched"
    assert unmatched_detections == [], "Detections have gone unmatched"



def test_all_inf_row_yields_unmatched_track():
    """
    Test with a track that has no viable detection
    Assert all end up in unmatched_tracks and does not crash
    """

    motion = ConstantVelocity2D(sigma_a=0.0)
    measurement = LinearPosition2D(sigma_pos=4.0)
    mtt = MultiTargetTracker(motion, measurement)

    # track 1: near the origin and will have a viable detection
    track1 = mtt._init_track(Detection(z=np.array([0.0, 0.0]), timestamp=0.0), timestamp=0.0)
    track1.predict(dt=0.1)

    # track 2: seeded somewhere nothing will be near, not a viable track
    track2 = mtt._init_track(Detection(z=np.array([1000.0, 1000.0]), timestamp=0.0), timestamp=0.0)
    track2.predict(dt=0.1)

    # only one detection given, near track1, so nothing to match with track2
    near_det = Detection(z=np.array([0.1, 0.1]), timestamp=0.1)

    gate = chi2_gate_threshold(dim_z=2, confidence=0.99)
    cost = build_cost_matrix([track1, track2], [near_det], gate)

    matches, unmatched_tracks, _ = hungarian_assign(cost)

    assert (0,0) in matches, "Track 1 not a match"
    assert 1 in unmatched_tracks, "Track 2's index not in unmatched track"

# To run: pytest tests/test_association.py -v
