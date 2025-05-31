import cv2
import base64
import time
import ollama

PROMPT = """
You are a robot that can see using a camera.
Your tasks is to move around the world.
DO NOT collide with any objects.
Stay away from walls.
Strictly only output one of the following commands:
'move forward', 'turn left', 'turn right', 'stop'
"""

MODEL_NAME = "qwen2.5vl:3b"
CAPTURE_DEVICE = 0          # change to URL / file if needed
FRAME_INTERVAL_SEC = 1.0    # send one frame per second to the model

def frame_to_b64(frame):
    """Encode a BGR OpenCV frame to base64-encoded JPEG string."""
    _, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def main():
    cap = cv2.VideoCapture(CAPTURE_DEVICE)
    if not cap.isOpened():
        print("Error: cannot access camera / video stream.")
        return

    print(f"Using model: {MODEL_NAME}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame, exiting.")
                break

            b64_img = frame_to_b64(frame)
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": PROMPT,
                    "images": [b64_img],
                }],
                stream=False,
            )
            action = response["message"]["content"].strip().lower()
            print(f"Action: {action}")

            time.sleep(FRAME_INTERVAL_SEC)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()

if __name__ == "__main__":
    main()
