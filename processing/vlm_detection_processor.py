from e2e_ad.processing.processing_module import ProcessingModule
from e2e_ad.data.sensor_data import SensorData

class VmlDetectionProcessor(ProcessingModule):
    def __init__(self, detector):
        self.detector = detector
        if not self.detector:
            print("[Warning] Detector module not available.", flush=True)

    def process(self, sensor_data: SensorData):
        if not self.detector:
            return sensor_data
        
        direction = self.detector.process(sensor_data.left_frame)

        print(f"[DEBUG] direction: {direction}", flush=True)

        sensor_data.vlm_direction = direction
        return sensor_data
