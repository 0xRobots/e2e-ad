from e2e_ad.navigation.navigation_strategy import NavigationStrategy
from e2e_ad.data.sensor_data import SensorData

class PotentialFieldStrategy(NavigationStrategy):
    def decide(self, sensor_data: SensorData) -> tuple[float, float]:
        # TODO: Implement potential field logic.
        return 1.0, 1.0