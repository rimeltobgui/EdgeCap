import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import cv2
import time

# Load BLIP model (CPU only)
device = torch.device("cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Capture from Pi camera using OpenCV
def capture_image_from_camera(save_path="captured.jpg"):
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


# Main captioning function
def generate_caption_from_camera():
    image = capture_image_from_camera()  # image is a valid PIL RGB image
    inputs = processor(images=image, return_tensors="pt").to(device)

    print("Running BLIP captioning...")
    start = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=30)
    end = time.time()

    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    print("\n===== BLIP Result =====")
    print(f"Caption: {caption}")
    print(f"Inference Time: {end - start:.2f} seconds")


# Run it
generate_caption_from_camera()
