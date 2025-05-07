import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import time

# Load BLIP model (CPU-only for Raspberry Pi)
device = torch.device("cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def caption_from_file(image_path):
    # Load and convert image
    image = Image.open(image_path).convert("RGB")

    # Preprocess and run inference
    inputs = processor(images=image, return_tensors="pt").to(device)

    print("Running BLIP captioning...")
    start = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=30)
    end = time.time()

    # Decode output
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    inference_time = end - start

    print("\n===== BLIP Caption Result =====")
    print(f"Caption: {caption}")
    print(f"Inference Time: {inference_time:.2f} seconds")

# Run it on your specific image
caption_from_file("/home/ubuntu/livingroom.jpg")
