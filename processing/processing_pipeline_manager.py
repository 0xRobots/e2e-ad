from e2e_ad.data.sensor_data import SensorData
from e2e_ad.data.sensor_data_hub import SensorDataHub

class ProcessingPipelineManager:
    def __init__(self, sensor_data_hub: SensorDataHub, processing_modules=None):
        self.sensor_data_hub = sensor_data_hub
        self.processing_modules = processing_modules or []

    def register_module(self, module):
        """Dynamically add a processing module."""
        self.processing_modules.append(module)

    def process_and_update(self, left_frame, right_frame):
        sensor_data = SensorData(left_frame=left_frame, right_frame=right_frame)

        for module in self.processing_modules:
            sensor_data = module.process(sensor_data)

        self.sensor_data_hub.update(sensor_data)
        return sensor_data
