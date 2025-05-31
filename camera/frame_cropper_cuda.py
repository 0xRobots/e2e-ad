import json
import cv2
import numpy as np
import threading
from typing import Tuple, Optional

class FrameCropper:
    def __init__(self, crop_path: str):
        """Initialize the FrameCropper with a configuration file path and GPU resources.
        
        Args:
            crop_path (str): Path to the JSON configuration file containing cropping parameters
        """
        self.config = self._load_config(crop_path)
        self.cuda_stream = cv2.cuda.Stream()
        self._lock = threading.Lock()
        
        # Pre-allocate GPU memory for input frames
        self.left_gpu = cv2.cuda_GpuMat()
        self.right_gpu = cv2.cuda_GpuMat()
        
        # Calculate ROI parameters once
        self.left_roi = (
            self.config["left_crop_x"],
            self.config["left_crop_y"],
            self.config["crop_width"],
            self.config["crop_height"]
        )
        self.right_roi = (
            self.config["right_crop_x"],
            self.config["right_crop_y"],
            self.config["crop_width"],
            self.config["crop_height"]
        )
    
    def _load_config(self, config_path: str) -> dict:
        """Load cropping parameters from a JSON file.
        
        Args:
            config_path (str): Path to the configuration file
            
        Returns:
            dict: Loaded configuration parameters
        """
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def crop_frames(self, left_frame: np.ndarray, right_frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Crop both left and right frames using GPU acceleration.
        
        Args:
            left_frame (np.ndarray): Left camera frame
            right_frame (np.ndarray): Right camera frame
            
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: Cropped left and right frames
        """
        if left_frame is None or right_frame is None:
            return None, None
            
        with self._lock:
            try:
                # Upload frames to GPU
                self.left_gpu.upload(left_frame, self.cuda_stream)
                self.right_gpu.upload(right_frame, self.cuda_stream)
                
                # Create GPU ROIs
                left_cropped_gpu = self.left_gpu.roi(*self.left_roi)
                right_cropped_gpu = self.right_gpu.roi(*self.right_roi)
                
                # Download results
                left_cropped = left_cropped_gpu.download(self.cuda_stream)
                right_cropped = right_cropped_gpu.download(self.cuda_stream)
                
                # Synchronize the CUDA stream
                self.cuda_stream.waitForCompletion()
                
                return left_cropped, right_cropped
                
            except cv2.error as e:
                print(f"CUDA error during frame cropping: {e}")
                return None, None
    
    def cleanup(self):
        """Release GPU resources."""
        with self._lock:
            self.left_gpu.release()
            self.right_gpu.release()
            # The cuda_stream is automatically released when the object is destroyed
