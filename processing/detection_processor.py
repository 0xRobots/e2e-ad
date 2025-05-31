from e2e_ad.processing.processing_module import ProcessingModule
from e2e_ad.data.sensor_data import SensorData

class DetectionProcessor(ProcessingModule):
    def __init__(self, detector):
        self.detector = detector
        if not self.detector:
            print("[Warning] Detector module not available.")

    def process(self, sensor_data: SensorData):
        if not self.detector:
            return sensor_data
        sensor_data.left_detections = self.detector.detect(sensor_data.left_frame, 'left')
        sensor_data.right_detections = self.detector.detect(sensor_data.right_frame, 'right')
        return sensor_data
