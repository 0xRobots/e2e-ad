from e2e_ad.navigation.navigation_strategy import NavigationStrategy
from e2e_ad.data.sensor_data import SensorData

class LocalMappingStrategy(NavigationStrategy):
    def decide(self, sensor_data: SensorData) -> tuple[float, float]:
        # TODO: Implement local mapping and lightweight planning.
        return 1.0, 1.0