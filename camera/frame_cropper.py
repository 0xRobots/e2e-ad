import json
import numpy as np
import threading
from typing import Tuple, Optional

class FrameCropper:
    def __init__(self, crop_path: str):
        """Initialize the FrameCropper with optimized parameters.
        
        Args:
            crop_path (str): Path to the JSON configuration file
        """
        self.config = self._load_config(crop_path)
        self._lock = threading.Lock()
        
        # Pre-calculate slicing indices for better performance
        self.left_slice = np.s_[
            self.config["left_crop_y"]:self.config["left_crop_y"] + self.config["crop_height"],
            self.config["left_crop_x"]:self.config["left_crop_x"] + self.config["crop_width"]
        ]
        self.right_slice = np.s_[
            self.config["right_crop_y"]:self.config["right_crop_y"] + self.config["crop_height"],
            self.config["right_crop_x"]:self.config["right_crop_x"] + self.config["crop_width"]
        ]
        
        # Pre-allocate memory for output frames
        self.left_output = None
        self.right_output = None
        
    def _load_config(self, config_path: str) -> dict:
        """Load cropping parameters from a JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def crop_frames(self, left_frame: np.ndarray, right_frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Crop frames using optimized CPU operations.
        
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
                # Initialize or update output arrays if needed
                if (self.left_output is None or 
                    self.left_output.shape != (self.config["crop_height"], self.config["crop_width"], left_frame.shape[2])):
                    self.left_output = np.empty((self.config["crop_height"], 
                                               self.config["crop_width"], 
                                               left_frame.shape[2]), 
                                              dtype=left_frame.dtype)
                    self.right_output = np.empty_like(self.left_output)
                
                # Use numpy's optimized view to crop without copying
                left_view = left_frame[self.left_slice]
                right_view = right_frame[self.right_slice]
                
                # Copy into pre-allocated arrays
                np.copyto(self.left_output, left_view)
                np.copyto(self.right_output, right_view)
                
                return self.left_output, self.right_output
                
            except Exception as e:
                print(f"Error during frame cropping: {e}")
                return None, None
    
    def cleanup(self):
        """Release resources."""
        with self._lock:
            self.left_output = None
            self.right_output = None
