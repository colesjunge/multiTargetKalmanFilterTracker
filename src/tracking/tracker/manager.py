import numpy as np

from ..types import Detection, FrameResult
from tracking.models.measurement import MeasurementModel 
from tracking.models.motion import MotionModel
from tracking.filters.kalman import KalmanFilter
from .track import Track
from tracking.association.assignment import hungarian_assign
from tracking.association.gating import chi2_gate_threshold, build_cost_matrix



class MultiTargetTracker():
    """
    Multi target tracking class

    Owns a set of live Track objects across all targets and frames.
    At each frame filter is advanced, new detections are incorporated, as well as the hit/miss cycle
    
    """

  
    def __init__(self, motion: MotionModel, measurement: MeasurementModel, n_confirm: int = 3, max_misses: int = 5, gate_confidence: float = .99):

        self.tracks: dict[int, Track] = {}
        self._next_id: int = 0
        self.motion = motion
        self.measurement = measurement
        self.n_confirm = n_confirm
        self.max_misses = max_misses
        self.gate_confidence = gate_confidence

    def step_cheated(self, detections_by_track_id: dict[int, Detection], dt: float, timestamp: float) -> FrameResult:
        """
        The cheat in this context if that the mapping is built rather than solved (which will be implemented later)

        in:  detections_by_track_id, dt, timestamp; dict[int, Detection] (maps an existing track's id to its own detection this frame), seconds, absolute current time in seconds
        out: FrameResult
        """

        # For all tracks advance predict
        for track in self.tracks.values():
            track.predict(dt)

        # For all tracks incorporate detections
        for track_id, det in detections_by_track_id.items():
            self.tracks[track_id].update(det)

        # For tracks not detected, mark missed
        for key in self.tracks.keys() - detections_by_track_id.keys():
            self.tracks[key].mark_missed()

        # If track hits are greater or equal to confirm, target confirmed
        for track in self.tracks.values():
            if track.hits >= self.n_confirm:
                track.confirmed = True

        # Build and return list of snapshots
        snapshots = []

        for track in self.tracks.values():
            snapshots.append(track.snapshot(timestamp))

        return FrameResult(timestamp=timestamp, snapshots=snapshots, n_detections=len(detections_by_track_id), assignments={})

    def step(self, detections: list[Detection], dt: float, timestamp: float) -> FrameResult:
            """
            Actual step association step which utilizes cost matrix, gating to keep plausbile pairs,
            and then uses Hungarian algorithm to solve resulting assingment (replaces step_cheated())

            In: detections, list[Detection]; this frame's raw, unlabeled detections
                dt; seconds since the last step
                timestamp; absolute current time (seconds)
            Out: FrameResult; snapshots of every live track (including those spawned within this frame),
                             n_detections, the track_ids (under assignments)
            """
    
            # For all tracks advance predict
            for track in self.tracks.values():
                track.predict(dt)
    
            # Build cost matrix
            track_ids_list = list(self.tracks.keys())
            tracks_list = list(self.tracks.values())
            gate = chi2_gate_threshold(self.measurement.dim_z, self.gate_confidence)
            cost = build_cost_matrix(tracks_list, detections, gate)
    
            # Get matches, unmatched tracks, unmatched detections
            matches, unmatched_track_idx, unmatched_det_idx = hungarian_assign(cost)
    
            # Update matched tracks
            for row, col in matches:
                track_id = track_ids_list[row]
                self.tracks[track_id].update(detections[col])

            # Mark matches
            for row in unmatched_track_idx:
                track_id = track_ids_list[row]
                self.tracks[track_id].mark_missed()

            # Confirm if hits exceed confirmation
            for track in self.tracks.values():
                if track.hits >= self.n_confirm:
                    track.confirmed = True

            # Create tentative tracks from unmatched detections
            for col in unmatched_det_idx:
                self._init_track(detections[col], timestamp)

            # Delete dead tracks
            for track_id in list(self.tracks.keys()):
                if self.tracks[track_id].is_dead(self.max_misses):
                    del self.tracks[track_id]

    
            # Build and return list of snapshots
            snapshots = [track.snapshot(timestamp) for track in self.tracks.values()]
            assignments = {track_ids_list[row]: col for row, col in matches}

            return FrameResult(timestamp=timestamp, snapshots=snapshots, n_detections=len(detections), assignments=assignments)

    def _init_track(self, det: Detection, timestamp: float) -> Track:
        """
        Initialize the track with set conditions

        in: det, timestamp; detection, absolute time track is created
        out: Track
        """

        x0 = np.array([det.z[0],
                        det.z[1],
                        0,
                        0])

        P0 = np.array([[10, 0, 0, 0], 
                [0, 12, 0, 0], 
                [0, 0, 34, 0], 
                [0, 0, 0, 35]]) # Initial uncertainty (random numbers I've chosen)

        kf = KalmanFilter(self.motion, self.measurement, x0, P0)

        new_id = self._next_id
        self._next_id += 1

        track = Track(new_id, kf, timestamp)

        self.tracks[new_id] = track

        return track



