from dataclasses import dataclass

@dataclass
class ObjectMetrics:
    class_name: str
    min_height_ratio: float
    max_height_ratio: float
    min_width_ratio: float
    max_width_ratio: float
    min_aspect_ratio: float
    max_aspect_ratio: float
    estimated_min_distance: float
    estimated_max_distance: float