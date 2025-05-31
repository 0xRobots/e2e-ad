import json
from e2e_ad.data.object_metrics import ObjectMetrics

class MetricsLoader:
    def __init__(self, json_path: str):
        """
        :param json_path: Path to the JSON file containing the metrics.
        """
        self.json_path = json_path

    def load_metrics(self) -> dict:
        """
        Loads metrics from the JSON file and returns a dictionary mapping labels
        to ObjectMetrics instances.
        """
        with open(self.json_path, "r") as f:
            metrics_data = json.load(f)

        metrics_dict = {}
        for label, data in metrics_data.items():
            metrics_dict[label] = ObjectMetrics(
                class_name=label,
                min_height_ratio=data["min_height_ratio"],
                max_height_ratio=data["max_height_ratio"],
                min_width_ratio=data["min_width_ratio"],
                max_width_ratio=data["max_width_ratio"],
                min_aspect_ratio=data["min_aspect_ratio"],
                max_aspect_ratio=data["max_aspect_ratio"],
                estimated_min_distance=data["estimated_min_distance"],
                estimated_max_distance=data["estimated_max_distance"]
            )
        return metrics_dict
