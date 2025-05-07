import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from PIL import Image
import requests
from io import BytesIO
import time
import nltk
#nltk.download("punkt")

# Step 1: Load BLIP model on CPU
device = torch.device("cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Step 2: Evaluation functions
def compute_cider(pred, refs):
    gts = {"0": [{"caption": r} for r in refs]}
    res = {"0": [{"caption": pred}]}
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)
    score, _ = Cider().compute_score(gts, res)
    return score

def compute_bleu(pred, refs):
    ref_tokens = [r.lower().split() for r in refs]
    pred_tokens = pred.lower().split()
    cc = SmoothingFunction()
    return sentence_bleu(ref_tokens, pred_tokens, smoothing_function=cc.method1)
    
meteor_scorer = Meteor()
def compute_meteor(pred, refs):
    gts = {"0": [{"caption": r} for r in refs]}  # each reference caption should be a dict
    res = {"0": [{"caption": pred}]}             # generated caption should also be a dict
    tokenizer_ = PTBTokenizer()
    gts = tokenizer_.tokenize(gts)
    res = tokenizer_.tokenize(res)
    score, _ = meteor_scorer.compute_score(gts, res)
    return score

# Step 3: Test images and references
image_urls = [
    "https://media.istockphoto.com/id/531347633/photo/soccer-players-in-action.jpg?s=612x612&w=0&k=20&c=mgufoWZknvbAnbnpqAambUll_NZwcrRrwlT6EVdMbtc=",
    "https://media.istockphoto.com/id/531347633/photo/soccer-players-in-action.jpg?s=612x612&w=0&k=20&c=mgufoWZknvbAnbnpqAambUll_NZwcrRrwlT6EVdMbtc=",
    "https://media.istockphoto.com/id/498766709/photo/holding-the-biscuit.jpg?s=612x612&w=0&k=20&c=jfaTRWvi55s5G5xC2vqmCLJDdEZCPI_cV1I1Y8t61D8="
]

reference_captions = [
    ["two soccer players on a field", "soccer players in action on the field"],
    ["two soccer players on a field", "soccer players in action on the field"],
    ["a golden retriever chewing on something", "a dog eating a treat"]
]
print(f"\n===== Blip Results =====")
# Step 4: Caption & Evaluate
for i, (url, refs) in enumerate(zip(image_urls, reference_captions)):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)
    start = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=30)
    end = time.time()

    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    inference_time = end - start

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
