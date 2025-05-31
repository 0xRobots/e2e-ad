class DistanceEstimator:
    def __init__(self, metrics):
        """
        Initialize with a dictionary mapping object labels to ObjectMetrics instances.
        
        :param metrics: dict mapping labels to ObjectMetrics instances.
        """
        self.metrics = metrics

    def process_detections(self, detections):
        """
        Process a list of detection dictionaries by computing and adding a 'distance' key to each.
        The camera height is obtained directly from each detection's dictionary.
        
        :param detections: List of detection dictionaries. Each dictionary must contain:
                           - 'bbox': [x1, y1, x2, y2]
                           - 'label': a string label for the detection.
                           - 'camera_height': the height of the camera frame.
        :return: The updated list of detection dictionaries.
        """
        for detection in detections:
            camera_height = detection.get("camera_height")
            label = detection.get("label")
            if label in self.metrics and camera_height:
                m = self.metrics[label]
                bbox = detection.get("bbox")
                detection_height = bbox[3] - bbox[1]
                observed_height_ratio = detection_height / camera_height

                if observed_height_ratio <= m.min_height_ratio:
                    detection["distance"] = m.estimated_max_distance
                elif observed_height_ratio >= m.max_height_ratio:
                    detection["distance"] = m.estimated_min_distance
                else:
                    detection["distance"] = m.estimated_min_distance + (
                        m.estimated_max_distance - m.estimated_min_distance
                    ) * (1 / observed_height_ratio - 1 / m.max_height_ratio) / (
                        1 / m.min_height_ratio - 1 / m.max_height_ratio
                    )
            else:
                detection["distance"] = None
        return detections
