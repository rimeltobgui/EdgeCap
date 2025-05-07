import cv2
from PIL import Image
import time
import open_clip
import torch

device = torch.device("cpu")

class TinyCLIPCaptioner:
    def __init__(self,
                 arch='TinyCLIP-ViT-39M-16-Text-19M',
                 pretrained='YFCC15M',
                 device=device):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=arch,
            pretrained=pretrained
        )
        self.model = self.model.to(device)
        self.tokenizer = open_clip.get_tokenizer(arch)

        self.candidate_prompts = [
            # Classroom prompts
            "a teacher writing on a whiteboard",
            "students raising their hands",
            "a projector displaying a presentation",
            "a classroom full of desks and chairs",
            "children listening attentively to a lesson",
            "a chalkboard covered in math equations",
            "a student giving a class presentation",
            "a teacher explaining a concept",
            "students reading textbooks",
            "laptops open on classroom desks",
            "a science experiment on the table",
            "a group discussion in class",
            "classroom with posters on the wall",
            "students wearing school uniforms",
            "a classroom with natural light",
            "a teacher holding a book",
            "students working in groups",
            "a test being taken in silence",
            "a student drawing on the board",
            "notebooks and pens on desks",
            "a classroom with colorful decorations",
            "students typing on computers",
            "backpacks hung on chairs",
            "a world map hanging on the wall",
            "a teacher helping a student individually",
        
            # Living room prompts
            "a couch in a cozy living room",
            "a family watching TV",
            "a cat sleeping on the couch",
            "a coffee table with books",
            "a television mounted on the wall",
            "a living room with large windows",
            "a fireplace and a rug",
            "someone reading on a sofa",
            "a dog playing with a toy indoors",
            "a plant in the corner of the room",
            "a person lounging with a laptop",
            "children playing in the living room",
            "a cluttered but warm living space",
            "a modern living room with artwork",
            "a person drinking coffee on the couch",
            "a game console on the table",
            "a couple sitting together watching TV",
            "a child drawing on the floor",
            "a well-lit living room interior",
            "a TV remote on the coffee table",
            "a blanket draped over a sofa",
            "decorative pillows on a couch",
            "a dog looking out the window",
            "a cozy living room at night",
            "books stacked on a shelf",
        
            # Tea cups and stuffed animals
            "a tea cup on a wooden table",
            "a tea cup and saucer near a window",
            "a person holding a tea cup",
            "a tea pot and cups set on a tray",
            "a tea cup with steam rising from it",
            "a tea cup beside an open book",
            "a child holding a stuffed animal",
            "a teddy bear sitting on a couch",
            "a row of stuffed animals on a shelf",
            "a stuffed animal lying on a bed",
            "a stuffed bunny on a pillow",
            "a tea party set up with stuffed animals"]

    def generate_caption(self, image):
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = self.tokenizer(self.candidate_prompts).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits_per_image = 100.0 * image_features @ text_features.T
        probs = logits_per_image.softmax(dim=-1).squeeze(0).cpu().numpy()

        best_idx = probs.argmax()
        best_caption = self.candidate_prompts[best_idx]
        confidence = probs[best_idx]
        return best_caption, confidence

def capture_image(save_path="captured.jpg"):
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise IOError("Cannot open camera")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise IOError("Failed to capture image")

    print("Captured image shape:", frame.shape)

    if len(frame.shape) != 3 or frame.shape[2] != 3:
        print("Warning: Unexpected frame shape, reshaping to RGB format.")
        frame = frame.reshape((480, 640, 3))

    success = cv2.imwrite(save_path, frame)
    if not success:
        raise IOError(f"Failed to save image to {save_path}")
    print(f"Image saved to {save_path}")

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return image

# Main process
tinyclip_model = TinyCLIPCaptioner()

print("Capturing image from camera...")
img = capture_image()

print("Running TinyCLIP captioning...")
start = time.time()
caption, confidence = tinyclip_model.generate_caption(img)
end = time.time()

print("\n===== TinyCLIP LIVE TEST RESULT =====")
print(f"Caption: {caption}")
print(f"Confidence: {confidence:.4f}")
print(f"Inference time: {end - start:.4f} seconds")
