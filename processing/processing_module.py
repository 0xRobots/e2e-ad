from abc import ABC, abstractmethod
from e2e_ad.data.sensor_data import SensorData

class ProcessingModule(ABC):
    @abstractmethod
    def process(self, sensor_data: SensorData):
        """Process sensor data and return updated data"""
        pass
