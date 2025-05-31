class DeepSortTracker:
    def __init__(self, iou_threshold=0.3, max_missed=3):
        """
        A simple DeepSORT-like tracker that uses IoU for detection-to-track association.
        
        :param iou_threshold: Minimum IoU to consider a detection matching an existing track.
        :param max_missed: Maximum number of consecutive frames a track can be missed before removal.
        """
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_track_id = 0
        # Dictionary mapping camera_id to a list of active tracks.
        self.tracks = {}

    def compute_iou(self, bbox1, bbox2):
        """
        Compute Intersection over Union (IoU) for two bounding boxes.
        
        :param bbox1: [x1, y1, x2, y2]
        :param bbox2: [x1, y1, x2, y2]
        :return: IoU as a float.
        """
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

    def update(self, detections, camera_id):
        """
        Update tracks for a given camera using the current detections.
        This method assigns a 'track_id' to each detection.
        
        :param detections: List of detection dictionaries, each containing:
                           - 'bbox': [x1, y1, x2, y2]
                           - 'label': a string label.
                           Other keys (like 'confidence', 'distance', etc.) remain intact.
        :param camera_id: Identifier for the camera (e.g., 'left' or 'right').
        :return: The list of detections with a new 'track_id' key added.
        """
        # Initialize track list for this camera if not present.
        if camera_id not in self.tracks:
            self.tracks[camera_id] = []
        active_tracks = self.tracks[camera_id]
        # Create a helper list to mark which tracks have been updated.
        assigned = [False] * len(active_tracks)

        # Process each detection.
        for detection in detections:
            best_iou = 0.0
            best_idx = -1
            # Try to find the best matching track based on IoU and same label.
            for i, track in enumerate(active_tracks):
                if detection.get("label") != track.get("label"):
                    continue
                iou = self.compute_iou(detection["bbox"], track["bbox"])
                if iou > best_iou and not assigned[i]:
                    best_iou = iou
                    best_idx = i
            if best_idx != -1 and best_iou >= self.iou_threshold:
                # A matching track was found: update the track.
                track = active_tracks[best_idx]
                detection["track_id"] = track["track_id"]
                track["bbox"] = detection["bbox"]
                track["missed"] = 0  # Reset missed count.
                assigned[best_idx] = True
            else:
                # No suitable track was found: create a new track.
                detection["track_id"] = self.next_track_id
                new_track = {
                    "track_id": self.next_track_id,
                    "bbox": detection["bbox"],
                    "label": detection["label"],
                    "missed": 0
                }
                active_tracks.append(new_track)
                assigned.append(True)
                self.next_track_id += 1

        # Increase missed count for tracks not updated in this frame.
        new_active_tracks = []
        for i, track in enumerate(active_tracks):
            if assigned[i]:
                new_active_tracks.append(track)
            else:
                track["missed"] += 1
                if track["missed"] <= self.max_missed:
                    new_active_tracks.append(track)
        self.tracks[camera_id] = new_active_tracks

        return detections
