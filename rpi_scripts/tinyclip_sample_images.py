# -*- coding: utf-8 -*-

# ==============================================================================
#  1) Install Dependencies
# ==============================================================================
import nltk
nltk.download('punkt')  # might be needed for tokenization

# ==============================================================================
#  2) Force CPU Usage
# ==============================================================================
import os
import torch

device = torch.device("cpu")  # We'll explicitly place our models on CPU

import open_clip
from PIL import Image

class TinyCLIPCaptioner:
    """
    A minimal class that wraps open_clip for TinyCLIP inference.
    Instead of true caption generation, we pick the top match from
    a small set of candidate strings.
    """
    def __init__(
        self,
        arch = 'TinyCLIP-ViT-39M-16-Text-19M',
        pretrained='YFCC15M',
        device=device
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

        # For demonstration, we define some candidate strings that will act like "captions"
        # In a real system, you'd have a generative approach or a broader set of phrases.
        self.candidate_prompts = [
            "soccer players on a field",
            "a golden retriever eating something",
            "a dog chewing a treat",
            "a beautiful sunset",
            "a diagram of a network",
            "a cat sleeping on a bed",
        ]

    def generate_caption(self, image):
        """
        Encode the image and compare it against our candidate_prompts.
        Return the best matching prompt, its confidence score, and all prompt probabilities.
        """
        # Preprocess the incoming PIL Image
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
    
        # Tokenize the candidate texts
        text_tokens = self.tokenizer(self.candidate_prompts).to(self.device)
    
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)
    
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
        # Compute similarity and probabilities
        logits_per_image = 100.0 * image_features @ text_features.T
        probs = logits_per_image.softmax(dim=-1).squeeze(0).cpu().numpy()
    
        # Pick the highest-prob text
        best_idx = probs.argmax()
        best_caption = self.candidate_prompts[best_idx]
        confidence = probs[best_idx]
    
        # Optionally: print all probabilities
        print("\n--- Candidate Prompt Probabilities ---")
        for prompt, p in zip(self.candidate_prompts, probs):
            print(f"{prompt}: {p:.4f}")
        print("--------------------------------------\n")
    
        return best_caption, confidence

# ==============================================================================
#  4) Instantiate Models (CPU)
# ==============================================================================
tinyclip_model = TinyCLIPCaptioner()


# ==============================================================================
#  5) Prepare Some Test Images & Reference Captions
# ==============================================================================
import requests
from PIL import Image
from io import BytesIO

image_urls = [
    "https://media.istockphoto.com/id/531347633/photo/soccer-players-in-action.jpg?s=612x612&w=0&k=20&c=mgufoWZknvbAnbnpqAambUll_NZwcrRrwlT6EVdMbtc=",
    "https://media.istockphoto.com/id/498766709/photo/holding-the-biscuit.jpg?s=612x612&w=0&k=20&c=jfaTRWvi55s5G5xC2vqmCLJDdEZCPI_cV1I1Y8t61D8="
]
for url in image_urls:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")

reference_captions = [
    ["two soccer players on a field", "soccer players in action on the field"],
    ["a golden retriever chewing on something", "a dog eating a treat"]
]

# ==============================================================================
#  6) Generate Captions with Each Model
# ==============================================================================
import time

def generate_captions(model, urls):
    captions = []
    for url in urls:
        resp = requests.get(url)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        start = time.time()
        caption, confidence = model.generate_caption(img)
        end = time.time()
        elapsed = end - start
        captions.append((caption, confidence, elapsed))
    return captions


tinyclip_captions = generate_captions(tinyclip_model, image_urls)
for i, (caption, conf, t) in enumerate(tinyclip_captions):
    print(f"Image {i+1} caption: {caption}")
    print(f"  Confidence: {conf:.4f}")
    print(f"  Inference time: {t:.4f}s\n")

# =============== Metrics Functions ===============


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

cc = SmoothingFunction()
meteor_scorer = Meteor()
tokenizer_ = PTBTokenizer()
cider_scorer = Cider()

def compute_bleu(pred, refs):
    ref_tokens = [r.lower().split() for r in refs]
    pred_tokens = pred.lower().split()
    return sentence_bleu(ref_tokens, pred_tokens, smoothing_function=cc.method1)

def compute_cider(pred, refs):
    gts = {"0": [{"caption": r} for r in refs]}
    res = {"0": [{"caption": pred}]}
    gts = tokenizer_.tokenize(gts)
    res = tokenizer_.tokenize(res)
    score, _ = cider_scorer.compute_score(gts, res)
    return score

def compute_meteor(pred, refs):
    gts = {"0": refs}
    res = {"0": [pred]}
    score, _ = meteor_scorer.compute_score(gts, res)
    return score

# === Per-image evaluation ===
print("===== TinyCLIP per-image Evaluation Results =====")
for i, (caption, _, _) in enumerate(tinyclip_captions):
    refs = reference_captions[i]

    bleu = compute_bleu(caption, refs)
    cider = compute_cider(caption, refs)
    meteor = compute_meteor(caption, refs)

    print(f"\nImage {i+1}")
    print(f"Generated Caption: {caption}")
    print(f"Reference Captions: {refs}")
    print(f"BLEU:   {bleu:.4f}")
    print(f"CIDEr:  {cider:.4f}")
    print(f"METEOR: {meteor:.4f}")
