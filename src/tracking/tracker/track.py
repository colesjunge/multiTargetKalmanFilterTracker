import numpy as np
from tracking.filters.base import BayesFilter
from ..types import Detection, TrackSnapshot


class Track():
    """
    Tracking wrapper for each target's filter and will be used within MultiTargetTracker 
    """

    def __init__(self, track_id: int, filt: BayesFilter, timestamp: float):

        self.track_id = track_id
        self.filter = filt

        self.hits = 1   # frames with a matched detection (track is born after 1 hit)
        self.misses = 0 # consecutive frames with no matched detection (resets to 0 on any update)
        self.age = 1    # total frames alive (regardless of hits/misses)
        self.confirmed = False  # promoted True by MultiTargetTracker once hits >= n_confirm

    def predict(self, dt: float) -> None:
        """ Advances filter's predict and increments age
        In: dt seconds
        Out: None; Mutates self.filter, self.age
        """
        self.filter.predict(dt)
        self.age += 1

    def update(self, det: Detection) -> None:
        """ Incorporates a matched detection into filter, resets consecutive misses, and increments hits
        In: det (dim_z,)
        Out: None; Mutates self.filter, self.misses, self.hits
        """
        self.filter.update(det.z)
        self.misses = 0
        self.hits += 1

    def mark_missed(self) -> None:
        """ Records no detection was matched to track in current frame
        In: None;
        Out: None; Mutates self.misses
        """
        self.misses += 1

    def is_dead(self, max_misses: int) -> bool:
        """ Records whether this track has gone undetected for too many consecutive frames
        In: max_misses; threshold of misses
        Out: Bool
        """
        return self.misses >= max_misses

    def snapshot(self, timestamp: float) -> TrackSnapshot:
        """ Creates immutable record of track's current state
        In: timestamp; this frames current time
        Out: TrackSnapshot
        """
        return TrackSnapshot(track_id=self.track_id, 
                             x=self.filter.x, 
                             P=self.filter.P, 
                             timestamp=timestamp, 
                             confirmed=self.confirmed)