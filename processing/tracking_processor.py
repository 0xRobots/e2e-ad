from e2e_ad.processing.processing_module import ProcessingModule
from e2e_ad.data.sensor_data import SensorData

class TrackingProcessor(ProcessingModule):
    def __init__(self, tracker):
        self.tracker = tracker
        if not self.tracker:
            print("[Warning] Tracker module not available.", flush=True)

    def process(self, sensor_data: SensorData):
        if not self.tracker:
            return sensor_data
        if not sensor_data.left_detections or not sensor_data.right_detections:
            print("[Warning] Tracking module requires detections but none were found.", flush=True)
            return sensor_data
        sensor_data.left_tracking = self.tracker.update(sensor_data.left_detections, 'left')
        sensor_data.right_tracking = self.tracker.update(sensor_data.right_detections, 'right')
        return sensor_data
