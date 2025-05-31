import threading
import time

class AutonomousNavigator:
    def __init__(self, sensor_data_hub, ws_client, strategy, decision_interval=0.2):
        """
        :param sensor_data_hub: Shared hub from which to retrieve the latest SensorData.
        :param ws_client: Instance of WebSocketClient.
        :param strategy: An instance of NavigationStrategy.
        :param decision_interval: Time (in seconds) between decision updates.
        """
        self.sensor_data_hub = sensor_data_hub
        self.ws_client = ws_client
        self.strategy = strategy
        self.decision_interval = decision_interval
        self.running = True
        self.enabled = False  # Autonomous driving is off by default.
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while self.running:
            sensor_data = self.sensor_data_hub.get_latest()
            if sensor_data is None:
                time.sleep(self.decision_interval)
                continue

            left_speed, right_speed = self.strategy.decide(sensor_data)
            if self.enabled:
                self.ws_client.send_command(left_speed, right_speed)
            time.sleep(self.decision_interval)

    def stop(self):
        self.running = False
        self.thread.join()
