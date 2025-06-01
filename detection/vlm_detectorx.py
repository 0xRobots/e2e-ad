from transformers import AutoModelForCausalLM, AutoTokenizer
import cv2
import numpy as np
from PIL import Image
from e2e_ad.data.sensor_data import SensorData

class VlmDetector2:
    def __init__(self, model_name="moondream/moondream-2b-2025-04-14-4bit"):
        """
        Initialize the VLM processor with Moondream model.
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map={"": "cuda"}
        )
        # Optional: Compile model for faster inference
        self.model.model.compile()
        
        self.prompt = """
        You are a robot that can see using a camera.
        Your task is to move around the world.
        DO NOT collide with any objects.
        Stay away from walls.
        Strictly only output one of the following commands:
        {forward}, {left}, {right}, {stop}
        """
        
        # Valid commands that the model can output
        self.valid_commands = {'forward', 'left', 'right', 'stop'}

    def _frame_to_pil(self, frame: np.ndarray) -> Image.Image:
        """Convert OpenCV BGR frame to PIL Image."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)

    def _validate_command(self, command: str) -> str:
        """Validate and normalize the model's output command."""
        command = command.strip().lower()
        if command not in self.valid_commands:
            return 'stop'  # Default to stop if invalid command
        return command

    def process(self, sensor_data: SensorData) -> SensorData:
        """
        Process frames using Moondream and update sensor data with navigation direction.
        Uses the left frame for decision making.
        """
        if sensor_data.left_frame is None:
            return sensor_data

        try:
            # Convert frame to PIL Image
            pil_image = self._frame_to_pil(sensor_data.left_frame)
            
            # Get model's response
            response = self.model.query(pil_image, self.prompt)
            command = self._validate_command(response["answer"])
            
            # Update sensor data with the direction
            sensor_data.vlm_direction = command
            
        except Exception as e:
            print(f"Error in VLM processing: {e}")
            sensor_data.vlm_direction = 'stop'  # Default to stop on error
            
        return sensor_data

    def cleanup(self):
        """Clean up model resources."""
        if hasattr(self, 'model'):
            del self.model 