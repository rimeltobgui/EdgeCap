from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import time

import nltk
nltk.download('punkt')

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

# === Load BLIP model ===
device = torch.device("cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast="True")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.to(device)

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

    # Generate caption
    inputs = processor(images=image, return_tensors="pt").to(device)
    start_time = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=30)
    end_time = time.time()

    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    inference_time = end_time - start_time

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
