from e2e_ad.processing.processing_module import ProcessingModule
from e2e_ad.data.sensor_data import SensorData

class DepthEstimationProcessor(ProcessingModule):
    def __init__(self, depth_estimator):
        self.depth_estimator = depth_estimator

    def process(self, sensor_data: SensorData):
        if not self.depth_estimator:
            print("[Warning] Depth estimation module not available.")
            return sensor_data
        sensor_data.depth_map = self.depth_estimator.estimate_depth(sensor_data.left_frame, sensor_data.right_frame)
        return sensor_data
