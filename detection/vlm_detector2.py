import cv2
import numpy as np
from PIL import Image
import tempfile
import ollama


class VlmDetector2:
    def __init__(self, model_name="moondream"):
        """
        Initialize the VLM processor using Ollama with Moondream model.
        """
        self.model_name = model_name
        self.prompt = (
            "You are a robot that can see using a camera.\n"
            "Your task is to move around the world.\n"
            "DO NOT collide with any objects.\n"
            "Stay away from walls.\n"
            "Strictly only output one of the following commands:\n"
            "{forward}, {left}, {right}, {stop}"
        )
        self.valid_commands = {"forward", "left", "right", "stop"}

    def _frame_to_pil(self, frame: np.ndarray) -> Image.Image:
        """Convert OpenCV BGR frame to PIL Image."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)

    def _validate_command(self, command: str) -> str:
        """Validate and normalize the model's output command."""
        command = command.strip().lower()
        return command if command in self.valid_commands else "stop"

    def _query_ollama(self, image: Image.Image) -> str:
        """Send image and prompt to Ollama and get a valid command."""
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
            image.save(temp_file.name, format="JPEG")
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": self.prompt,
                        "images": [temp_file.name],
                    }
                ],
            )
        return response["message"]["content"]

    def process(self, frame):
        """
        Process frames using Ollama and update sensor data with navigation direction.
        Uses the left frame for decision making.
        """
        if frame is None:
            return "stop"

        try:
            pil_image = self._frame_to_pil(frame)
            response = self._query_ollama(pil_image)
            action = self._validate_command(response)
            print(f"[DEBUG] VLM Response: {response}", flush=True)
            return action
        except Exception as e:
            print(f"Error in VLM processing: {e}", flush=True)
            action = "stop"

        return action

    def cleanup(self):
        """Clean up resources (no-op for Ollama)."""
        pass
