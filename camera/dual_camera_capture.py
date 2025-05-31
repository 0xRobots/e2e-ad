import cv2
import time
import threading
import os

class DualCameraCapture:
    def __init__(self, stream1_url, stream2_url):
        self.stream_urls = [stream1_url, stream2_url]
        self.frames = {i: None for i in range(2)}
        self.running = True
        self.threads = []
        
        # Check if GStreamer is available
        self.use_gstreamer = self._check_gstreamer_available()
        if self.use_gstreamer:
            print("Using GStreamer backend for video capture")
        else:
            print("Using FFMPEG backend for video capture")

    def _check_gstreamer_available(self):
        # Create a simple test pipeline to check if GStreamer is available
        test_pipeline = "videotestsrc num-buffers=1 ! appsink"
        test_cap = cv2.VideoCapture(test_pipeline, cv2.CAP_GSTREAMER)
        is_available = test_cap.isOpened()
        test_cap.release()
        return is_available

    def open_camera_stream(self, cam_index):
        attempts = 0
        cap = None
        url = self.stream_urls[cam_index]
        
        while attempts < 3:
            if self.use_gstreamer:
                # Use GStreamer pipeline
                pipeline = f"rtspsrc location={url} protocols=tcp latency=0 ! rtph264depay ! h264parse ! nvh264dec ! videoconvert ! appsink"
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            else:
                # Use FFMPEG
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
            if cap.isOpened():
                print(f"Camera {cam_index} stream opened successfully using {'GStreamer' if self.use_gstreamer else 'FFMPEG'}.")
                return cap
            else:
                attempts += 1
                backend = "GStreamer" if self.use_gstreamer else "FFMPEG"
                print(f"Attempt {attempts} to open camera {cam_index} stream at {url} with {backend} failed, retrying in 5 seconds...")
                time.sleep(5)
        
        return None

    def capture_frames(self, cam_index):
        cap = self.open_camera_stream(cam_index)
        if cap is None:
            print(f"Failed to open camera {cam_index} stream after 3 attempts. Shutting down.")
            os._exit(1)

        while self.running:
            ret, frame = cap.read()
            if ret:
                # Save a copy of the raw frame
                self.frames[cam_index] = frame.copy()
            else:
                print(f"Failed to grab frame from cam{cam_index}")
                time.sleep(0.1)
        cap.release()

    def start(self):
        for i in range(len(self.stream_urls)):
            thread = threading.Thread(target=self.capture_frames, args=(i,), daemon=True)
            thread.start()
            self.threads.append(thread)

    def stop(self):
        self.running = False
        for thread in self.threads:
            thread.join()