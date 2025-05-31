import cv2

class FrameVisualizer:
    def __init__(self):
        # You can initialize default fonts, colors, or any parameters here.
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.robot_color = (255, 0, 0)
        self.wall_color = (0, 255, 0)
        self.corner_color = (0, 0, 255)
        self.track_color = (0, 255, 255)

    def draw_enriched_frame(self, frame, detections, tracking_objects=None):
        """
        Draw bounding boxes, labels, distance info, and tracking IDs on the frame.
        
        :param frame: The image frame as a NumPy array.
        :param detections: List of detection dictionaries, where each dictionary should
                           have at least the keys "bbox", "label", "confidence", "camera_height",
                           and optionally "distance" and "track_id".
        :param tracking_objects: (Optional) List of tracking dictionaries with keys "track_id",
                                 "position", "velocity", etc.
        :return: The annotated frame.
        """
        # Draw detection bounding boxes.
        for det in detections:
            bbox = det.get("bbox")
            label = det.get("label", "object")
            confidence = det.get("confidence", 0)
            distance = det.get("distance", None)
            track_id = det.get("track_id", None)
            
            if distance is not None:
                display_text = f"{label} {confidence:.2f}%: {distance:.2f}m"
            else:
                display_text = f"{label} {confidence:.2f}%: ??m"
            
            # Convert coordinates to integers.
            x1, y1, x2, y2 = map(int, bbox)
            # Choose a color based on label.
            if label == "robot":
                color = self.robot_color
            elif label == "wall_corner":
                color = self.corner_color
            else:
                color = self.wall_color
            
            # Draw the bounding box and label.
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, display_text, (x1, max(y1 - 10, 0)), self.font, 1, color, 1)

            # Append tracking ID if available.
            if track_id is not None:
                tracking_text = f" ID:{track_id}"
                cv2.putText(frame, tracking_text, (x1+25, y1+25), self.font, 1, self.track_color, 1)
        
        # Draw additional tracking objects if provided.
        if tracking_objects is not None:
            for track in tracking_objects:
                # Assume the first two coordinates of the 'position' represent pixel coordinates.
                pos = track.get("position", [0, 0])
                x = int(pos[0])
                y = int(pos[1])
                t_id = track.get("track_id", -1)
                
                # Draw a small circle at the track position.
                cv2.circle(frame, (x, y), 5, self.track_color, -1)
                # Label the track with its ID.
                cv2.putText(frame, f"ID:{t_id}", (x + 5, y - 5), self.font, 0.5, self.track_color, 1)
        return frame
