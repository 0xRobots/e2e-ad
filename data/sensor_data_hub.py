import threading

class SensorDataHub:
    def __init__(self):
        self._latest = None
        self._lock = threading.Lock()

    def update(self, sensor_data):
        """Atomically update the latest sensor data."""
        with self._lock:
            self._latest = sensor_data

    def get_latest(self):
        """Atomically retrieve the latest sensor data."""
        with self._lock:
            return self._latest
