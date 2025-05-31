import cv2
import numpy as np

class DualCameraRenderer:
    def __init__(self, window_name="Robot Navigation"):
        self.window_name = window_name

    def show(self, frames):
        """
        Combines the frames from multiple cameras and displays them.

        :param frames: Dictionary of frames, keyed by camera index.
        """
        processed_frames = []
        for i in sorted(frames.keys()):
            if frames[i] is None:
                continue
            frame_copy = frames[i].copy()
            processed_frames.append(frame_copy)
        if not processed_frames:
            return

        combined_frame = np.hstack(tuple(processed_frames))
        # Print only the shape of the combined frame to avoid flooding the console.
        #print("[DEBUG] got frames, combined shape:", combined_frame.shape, flush=True)

        cv2.imshow(self.window_name, combined_frame)
