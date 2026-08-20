from dataclasses import dataclass, field
import numpy as np

@dataclass
class Detection:
    # A single sensor return in one frame.
    z: np.ndarray   # shape (dim_z,); e.g. (px, py)
    timestamp: float    # seconds
    sensor_id: str = "radar"
    meta: dict = field(default_factory=dict)    # SNR, etc.

@dataclass
class TrackSnapshot:
    # Immutable record of one track's state at one frame
    track_id: int
    x: np.ndarray   # shape (dim_x,); posterior state estimate
    P: np.ndarray   # shape (dim_x, dim_x); posterior covariance
    timestamp: float
    confirmed: bool

@dataclass
class FrameResult:
    # Everything the tracker produced for one time step. (List of track snapshots)
    timestamp: float
    snapshots: list[TrackSnapshot]
    n_detections: int
    assignments: dict[int, int] # track_id -> detection index (this frame)

