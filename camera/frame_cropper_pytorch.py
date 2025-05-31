import json
import numpy as np
import torch
import threading
from typing import Tuple, Optional

class FrameCropper:
    def __init__(self, crop_path: str):
        """Initialize the FrameCropper with PyTorch GPU support.
        
        Args:
            crop_path (str): Path to the JSON configuration file
        """

        print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
        print(f"Current device: {torch.cuda.current_device()}", flush=True)
        print(f"Device name: {torch.cuda.get_device_name(0)}", flush=True)

        self.config = self._load_config(crop_path)
        self._lock = threading.Lock()
        
        # Set up CUDA device if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Pre-calculate slicing indices
        self.left_slice = (
            slice(self.config["left_crop_y"], self.config["left_crop_y"] + self.config["crop_height"]),
            slice(self.config["left_crop_x"], self.config["left_crop_x"] + self.config["crop_width"]),
            slice(None)  # Keep all channels
        )
        self.right_slice = (
            slice(self.config["right_crop_y"], self.config["right_crop_y"] + self.config["crop_height"]),
            slice(self.config["right_crop_x"], self.config["right_crop_x"] + self.config["crop_width"]),
            slice(None)  # Keep all channels
        )
        
    def _load_config(self, config_path: str) -> dict:
        """Load cropping parameters from a JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor on GPU."""
        # OpenCV images are in BGR format and numpy arrays
        # Convert to torch tensor and move to GPU if available
        return torch.from_numpy(frame).to(self.device)
    
    def _tensor_to_frame(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor back to numpy array."""
        # Move tensor to CPU and convert back to numpy
        return tensor.cpu().numpy()
    
    def crop_frames(self, left_frame: np.ndarray, right_frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Crop frames using PyTorch GPU acceleration.
        
        Args:
            left_frame (np.ndarray): Left camera frame
            right_frame (np.ndarray): Right camera frame
            
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: Cropped frames
        """
        if left_frame is None or right_frame is None:
            return None, None
            
        with self._lock:
            try:
                # Convert frames to tensors and move to GPU
                left_tensor = self._frame_to_tensor(left_frame)
                right_tensor = self._frame_to_tensor(right_frame)
                
                # Perform cropping on GPU
                left_cropped = left_tensor[self.left_slice]
                right_cropped = right_tensor[self.right_slice]
                
                # Convert back to numpy arrays
                left_result = self._tensor_to_frame(left_cropped)
                right_result = self._tensor_to_frame(right_cropped)
                
                # Ensure contiguous arrays
                return np.ascontiguousarray(left_result), np.ascontiguousarray(right_result)
                
            except Exception as e:
                print(f"Error during frame cropping: {e}")
                return None, None
    
    def cleanup(self):
        """Release GPU memory."""
        with self._lock:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
