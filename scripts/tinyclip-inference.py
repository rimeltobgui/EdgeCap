# === Import required libraries ===
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from PIL import Image

import time
from torchvision import transforms
import open_clip

from tqdm import tqdm
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import nltk

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


# === Set device for inference ===
device = torch.device("cpu")

# === Load TinyCLIP-ViT-39M-16 model ===
class TinyCLIPCaptioner:

    def __init__(
        self,
        arch = 'TinyCLIP-ViT-39M-16-Text-19M',
        pretrained='YFCC15M',
        device='cpu'
    ):
        self.device = device

        # Create model & transforms from open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=arch,
            pretrained=pretrained
        )
        self.model = self.model.to(device)

        # Grab the tokenizer
        self.tokenizer = open_clip.get_tokenizer(arch)

        self.candidate_prompts = [
            "soccer players on a field",
            "a dog chewing a treat",
            "a beautiful sunset",
            "a diagram of a network",
            "a cat sleeping on a bed",
            "A group of students attending a lecture in a university classroom",
            "Students taking notes while a professor teaches at the front",
            "A busy college classroom filled with laptops and notebooks",
            "A professor presenting slides to a class of university students",
            "University students seated in rows, listening to a lecture",
            "A classroom discussion between students and a professor",
            "College students focused on a presentation in a lecture hall",
            "A chalkboard filled with notes in a college classroom",
            "Students raising their hands to ask questions during class",
            "A modern classroom with digital screens and engaged students"
        ]

    def generate_caption(self, image):
        """
        Encode the image and compare it against our candidate_prompts.
        Return the best matching prompt as a pseudo 'caption.'
        """
        # Preprocess the incoming PIL Image
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # Tokenize the candidate texts
        text_tokens = self.tokenizer(self.candidate_prompts).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            # Compute features
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Similarities & best match
        logits_per_image = 100.0 * image_features @ text_features.T
        probs = logits_per_image.softmax(dim=-1)

        # Pick the highest-prob text
        best_idx = probs.argmax(dim=-1).item()
        best_caption = self.candidate_prompts[best_idx]
        return best_caption
    

tinyclip_model = TinyCLIPCaptioner()

# === Metric scoring functions ===
cc = SmoothingFunction()
meteor_scorer = Meteor()

def compute_bleu(pred, refs):
    refs_tokenized = [r.lower().split() for r in refs]
    pred_tokenized = pred.lower().split()
    return sentence_bleu(refs_tokenized, pred_tokenized, smoothing_function=cc.method1)

def compute_cider(pred, refs):
    gts = {"0": [{"caption": r} for r in refs]}
    res = {"0": [{"caption": pred}]}
    tokenizer_ = PTBTokenizer()
    gts = tokenizer_.tokenize(gts)
    res = tokenizer_.tokenize(res)
    score, _ = Cider().compute_score(gts, res)
    return score

def compute_meteor(pred, refs):
    gts = {"0": [{"caption": r} for r in refs]}
    res = {"0": [{"caption": pred}]}
    tokenizer_ = PTBTokenizer()
    gts = tokenizer_.tokenize(gts)
    res = tokenizer_.tokenize(res)
    score, _ = meteor_scorer.compute_score(gts, res)
    return score

# === Image paths and ground truth captions ===
image_paths = [
    "test_images/soccer.jpg",
    "test_images/dog.jpg",
    "test_images/classroom.png"
]

reference_captions = [
    ["two soccer players on a field", "soccer players in action on the field"],
    ["a golden retriever chewing on something", "a dog eating a treat"],
    ["A group of people attentively working on laptops in a classroom.", "Students seated in rows, focused on a lecture or assignment."]
]

# === Caption and evaluate for each image ===
for i, (path, refs) in enumerate(zip(image_paths, reference_captions)):
    image = Image.open(path).convert("RGB")

    start_time = time.time()
    caption = tinyclip_model.generate_caption(image)
    end_time = time.time()

    inference_time = end_time - start_time
    cider = compute_cider(caption, refs)
    meteor = compute_meteor(caption, refs)
    bleu = compute_bleu(caption, refs)

    print(f"\n===== Image {i+1} =====")
    print(f"Generated Caption: {caption}")
    print(f"Reference Captions: {refs}")
    print(f"Inference Time: {inference_time:.4f} seconds")
    print(f"CIDEr Score:  {cider:.4f}")
    print(f"METEOR Score: {meteor:.4f}")
    print(f"BLEU Score:   {bleu:.4f}")
