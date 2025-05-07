import cv2
from PIL import Image
import numpy as np

def capture_image(save_path="captured.jpg"):
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    # Optional: explicitly request resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise IOError("Cannot open camera")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise IOError("Failed to capture image")

    print("Captured image shape:", frame.shape)

    # If grayscale or malformed, reshape/check
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        print("?? Unexpected frame shape, converting to RGB manually...")
        frame = frame.reshape((480, 640, 3))

    # Save and convert
    success = cv2.imwrite(save_path, frame)
    if not success:
        raise IOError(f"Failed to save image to {save_path}")
    print(f"? Image saved to {save_path}")

    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Run it
capture_image()
