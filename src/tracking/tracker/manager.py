import numpy as np

from ..types import Detection, FrameResult
from tracking.models.measurement import MeasurementModel 
from tracking.models.motion import MotionModel
from tracking.filters.kalman import KalmanFilter
from .track import Track



class MultiTargetTracker():
    """
    Multi target tracking class

    Owns a set of live Track objects across all targets and frames.
    At each frame filter is advanced, new detections are incorporated, as well as the hit/miss cycle
    
    """

  
    def __init__(self, motion: MotionModel, measurement: MeasurementModel, n_confirm: int = 3, max_misses: int = 5):

        self.tracks: dict[int, Track] = {}
        self._next_id: int = 0
        self.motion = motion
        self.measurement = measurement
        self.n_confirm = n_confirm
        self.max_misses = max_misses

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



