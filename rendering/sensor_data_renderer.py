class SensorDataRenderer:
    def __init__(self, renderer):
        self.renderer = renderer
        self.render_enriched = False

    def show(self, sensor_data):
        """
        Gets the frames from SensorData object and passes them to the renderer.

        :param sensor_data: SensorData object
        """
        if sensor_data is None:
            #print("[DEBUG] sensor data is NONE", flush=True)
            return

        if self.render_enriched:
            frames = {
                0: sensor_data.left_frame_visualized if sensor_data.left_frame_visualized is not None else sensor_data.left_frame,
                1: sensor_data.right_frame_visualized if sensor_data.right_frame_visualized is not None else sensor_data.right_frame,
            }
        else:
            frames = {
                0: sensor_data.left_frame,
                1: sensor_data.right_frame,
            }

        self.renderer.show(frames)
