from dataclasses import dataclass, field
from typing import List, Any
import numpy as np

@dataclass
class SensorData:
    left_frame: np.ndarray = None
    right_frame: np.ndarray = None
    left_frame_visualized: np.ndarray = None
    right_frame_visualized: np.ndarray = None
    left_detections: List[dict] = field(default_factory=list)
    right_detections: List[dict] = field(default_factory=list)
    left_tracking: List[Any] = field(default_factory=list)
    right_tracking: List[Any] = field(default_factory=list)
