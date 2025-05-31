from time import sleep
from e2e_ad.navigation.navigation_strategy import NavigationStrategy
from e2e_ad.data.sensor_data import SensorData

# Define critical thresholds.
SAFE_DISTANCE = 0.15  # Minimum safe distance in meters

class ReactiveBehaviorStrategy(NavigationStrategy):

    def decide(self, sensor_data: SensorData) -> tuple[float, float]:
        # Default: STOP the robot when no detections available or no decision can be made:
        left_cmd = 0.0
        right_cmd = 0.0

        # Extract detected distances
        left_distances = [det["distance"] for det in sensor_data.left_detections]
        right_distances = [det["distance"] for det in sensor_data.right_detections]

        # Get the closest obstacle on each side
        min_left_distance = min(left_distances, default=float('inf'))
        min_right_distance = min(right_distances, default=float('inf'))

        #print(f"[DEBUG] Min Left Distance: {min_left_distance}, Min Right Distance: {min_right_distance}", flush=True)

        # Check if an obstacle is too close
        obstacle_left = min_left_distance < SAFE_DISTANCE
        obstacle_right = min_right_distance < SAFE_DISTANCE

        if obstacle_left and obstacle_right:
            # Both sides blocked → Turn around
            #print("[DEBUG] Corner detected! Turning around...", flush=True)
            left_cmd = 0.7
            right_cmd = -0.7

        elif obstacle_left:
            # Obstacle on the left → Turn right
            #print("[DEBUG] Obstacle on the left! Turning right...", flush=True)
            left_cmd = 1.0
            right_cmd = 0.0

        elif obstacle_right:
            # Obstacle on the right → Turn left
            #print("[DEBUG] Obstacle on the right! Turning left...", flush=True)
            left_cmd = 0.0
            right_cmd = 1.0

        else:
            left_cmd = 1.0
            right_cmd = 1.0

        # for track in sensor_data.tracking_objects:
        #     state = track.get_state()
        #     print(f"[DEBUG] Track ID: {track.id}")
        #     print(f"Position: {state}")

        return left_cmd, right_cmd
