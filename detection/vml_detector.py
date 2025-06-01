import cv2
import numpy as np
from PIL import Image
import ollama
import tempfile
from e2e_ad.data.sensor_data_hub import SensorDataHub


VALID_COMMANDS = {"forward", "left", "right", "stop"}


def build_prompt() -> str:
    return (
        "You are a robot that can see using a camera.\n"
        "Your task is to move around the world.\n"
        "DO NOT collide with any objects.\n"
        "Stay away from walls.\n"
        "Strictly only output one of the following commands:\n"
        "{forward}, {left}, {right}, {stop}"
    )


def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR frame to a PIL RGB Image."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)


def get_robot_command(image: Image.Image, model_name: str = "qwen2.5vl:3b") -> str:
    """
    Sends a PIL image and prompt to the Ollama model and returns one valid command.
    """
    prompt_text = build_prompt()

    # Save the image temporarily (Ollama currently supports image file paths, not PIL objects)
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp:
        image.save(temp.name, format="JPEG")

        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt_text,
                    "images": [temp.name],  # Provide path to JPEG
                }
            ],
        )

    command = response["message"]["content"].strip().lower()

    if command not in VALID_COMMANDS:
        raise ValueError(
            f"Ollama returned invalid command '{command}'. "
            f"Expected one of {', '.join(sorted(VALID_COMMANDS))}."
        )
    return command


class VlmDetector:
    """
    Detector using Ollama with PIL Images instead of base64.
    """

    def __init__(self, model_name: str, hub: SensorDataHub):
        self.model_name = model_name
        self.hub = hub

    def detect(self, frame, camera_id: str):
        """
        Accepts an OpenCV frame, converts to PIL, and queries Ollama for movement command.
        """
        try:
            pil_image = _frame_to_pil(frame)
            action = get_robot_command(pil_image, model_name=self.model_name)
        except Exception as e:
            print(f"Ollama error: {e}")
            action = "stop"

        self.hub.update({"direction": action})
        return []
