from ultralytics import YOLO
from e2e_ad.config import CLASS_MAPPING

class YoloDetector:
    def __init__(self, model_path, detection_memory_size=5, memory_threshold=2, iou_threshold=0.3):
        """
        :param detection_memory_size: Number of frames to remember for temporal smoothing.
        :param memory_threshold: Minimum number of past frames in which a similar detection must appear.
        :param iou_threshold: IoU threshold to consider two detections as matching.
        """
        self.model = YOLO(model_path, task='detect')
        # Maintain separate detection histories per camera (e.g., 'left' and 'right')
        self.detection_history = {}
        self.detection_memory_size = detection_memory_size
        self.memory_threshold = memory_threshold
        self.iou_threshold = iou_threshold

    def predict(self, frame):
        """Run inference on a frame."""
        return self.model.predict(frame, stream=False, device='cuda', verbose=False)

    def extract_detections(self, frame, predictions):
        """
        Given a YOLO result as predictions, extract a list of dictionaries for each detection.
        Each dictionary contains:
        - label: the detected class label
        - bbox: [x1, y1, x2, y2] (in pixels)
        - confidence: confidence score (float)
        - camera_width, camera_height: for computing relative area
        """
        detections = []
        if not hasattr(predictions, 'boxes') or predictions.boxes is None:
            return detections

        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        for box in predictions.boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = CLASS_MAPPING.get(cls_id, str(cls_id))

            detections.append({
                "label": label,
                "bbox": xyxy,
                "confidence": conf,
                "camera_width": frame_width,
                "camera_height": frame_height
            })
        return detections

    @staticmethod
    def compute_iou(bbox1, bbox2):
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = area1 + area2 - intersection_area
        if union_area == 0:
            return 0.0
        return intersection_area / union_area

    def smooth_box_size(self, detection, history):
        """
        Smooth only the size of the bbox (width and height) by averaging over matching detections 
        in the history, while keeping the current center intact.
        """
        x1, y1, x2, y2 = detection["bbox"]
        center = [(x1 + x2) / 2, (y1 + y2) / 2]
        widths = []
        heights = []
        for frame in history:
            for det in frame:
                if det["label"] == detection["label"]:
                    if self.compute_iou(detection["bbox"], det["bbox"]) >= self.iou_threshold:
                        bw = det["bbox"][2] - det["bbox"][0]
                        bh = det["bbox"][3] - det["bbox"][1]
                        widths.append(bw)
                        heights.append(bh)
                        break  # Only one match per frame.
        if widths and heights:
            avg_width = sum(widths) / len(widths)
            avg_height = sum(heights) / len(heights)
            new_x1 = center[0] - avg_width / 2
            new_y1 = center[1] - avg_height / 2
            new_x2 = center[0] + avg_width / 2
            new_y2 = center[1] + avg_height / 2
            return [new_x1, new_y1, new_x2, new_y2]
        return detection["bbox"]

    def detect(self, frame, camera_id):
        """
        Detect objects in a given frame from a specific camera.
        :param frame: Input image frame.
        :param camera_id: Identifier for the camera (e.g., 'left' or 'right').
        """
        predictions = self.predict(frame)
        current_detections = self.extract_detections(frame, predictions[0])
        
        # Initialize history for camera if not present
        if camera_id not in self.detection_history:
            self.detection_history[camera_id] = []
        
        # Update detection history for this camera
        self.detection_history[camera_id].append(current_detections)
        if len(self.detection_history[camera_id]) > self.detection_memory_size:
            self.detection_history[camera_id].pop(0)
        
        # For initial frames, return current detections to bootstrap memory.
        if len(self.detection_history[camera_id]) < self.detection_memory_size:
            return current_detections
        
        # Filter detections based on temporal consistency for this camera.
        filtered_detections = []
        for det in current_detections:
            consistent_count = 0
            for past_frame in self.detection_history[camera_id][:-1]:
                for past_det in past_frame:
                    if past_det["label"] == det["label"]:
                        if self.compute_iou(det["bbox"], past_det["bbox"]) >= self.iou_threshold:
                            consistent_count += 1
                            break  # Only count one match per past frame.
            if consistent_count >= self.memory_threshold:
                # Only smooth the box size while keeping the current center.
                det["bbox"] = self.smooth_box_size(det, self.detection_history[camera_id])
                filtered_detections.append(det)
        return filtered_detections
