from e2e_ad.processing.processing_module import ProcessingModule
from e2e_ad.data.sensor_data import SensorData

class DistanceEstimationProcessor(ProcessingModule):
    def __init__(self, distance_estimator):
        """
        :param distance_estimator: An instance of DistanceEstimator.
        """
        self.distance_estimator = distance_estimator

    def process(self, sensor_data: SensorData):
        # Delegate processing to the DistanceEstimator for left detections.
        if sensor_data.left_detections:
            sensor_data.left_detections = self.distance_estimator.process_detections(
                sensor_data.left_detections
            )
        # Delegate processing for right detections.
        if sensor_data.right_detections:
            sensor_data.right_detections = self.distance_estimator.process_detections(
                sensor_data.right_detections
            )
        return sensor_data
