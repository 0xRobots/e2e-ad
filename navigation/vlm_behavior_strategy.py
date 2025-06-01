from e2e_ad.navigation.navigation_strategy import NavigationStrategy
from e2e_ad.data.sensor_data import SensorData

class VlmBehaviorStrategy(NavigationStrategy):
    def __init__(self):
        # Speed settings for different actions
        self.forward_speed = 1.0
        self.turn_speed = 1.0
        self.stop_speed = 0.0

    def decide(self, sensor_data: SensorData) -> tuple[float, float]:
        """
        Make navigation decisions based on VLM direction from sensor data.
        Returns (left_speed, right_speed) tuple.
        """

        if sensor_data.vlm_direction is None:
            return self.stop_speed, self.stop_speed

        direction = sensor_data.vlm_direction
        print(f"[DEBUG] vlm_direction: {direction}", flush=True)

        # Map VLM direction to motor speeds
        if sensor_data.vlm_direction == 'forward':
            return self.forward_speed, self.forward_speed
        elif sensor_data.vlm_direction == 'left':
            return -self.turn_speed, self.turn_speed
        elif sensor_data.vlm_direction == 'right':
            return self.turn_speed, -self.turn_speed
        else:  # 'stop' or unknown
            return self.stop_speed, self.stop_speed 