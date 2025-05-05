# === Import required libraries ===
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from PIL import Image
from IPython.display import display

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

# === MLP Mapper: maps CLIP embedding to GPT2-compatible prefix tokens ===
class MLPMapper(nn.Module):
    def __init__(self, clip_dim=512, gpt2_dim=768, prefix_length=10):
        super().__init__()
        self.prefix_length = prefix_length
        self.mapper = nn.Sequential(
            nn.Linear(clip_dim, gpt2_dim * prefix_length),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.mapper(x)
        batch_size = x.size(0)
        gpt2_dim = x.size(1) // self.prefix_length
        return x.view(batch_size, self.prefix_length, gpt2_dim)

# === Image preprocessing pipeline (ViT expected format) ===
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# === Full TinyCLIP-to-GPT2 captioning model ===
class TinyCLIPCap(nn.Module):
    def __init__(self, clip_model, gpt2_model, prefix_length=10):
        super().__init__()
        self.clip_model = clip_model
        self.gpt2 = gpt2_model
        self.prefix_length = prefix_length
        self.mapper = MLPMapper(
            clip_dim=clip_model.visual.output_dim,
            gpt2_dim=gpt2_model.config.n_embd,
            prefix_length=prefix_length
        )

    def forward(self, image_embedding):
        return self.mapper(image_embedding)

# === Set device for inference ===
device = "cpu"

# === Load CLIP ViT-B/32 model ===
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model.eval()

# === Load DistilGPT2 and tokenizer ===
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
gpt2 = GPT2LMHeadModel.from_pretrained("distilgpt2")
gpt2.eval()

# === Assemble full model ===
model = TinyCLIPCap(clip_model, gpt2)

# === Setup tokenizer pad token ===
tokenizer.pad_token = tokenizer.eos_token

# === Normalize input images ===
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # for RGB inputs, modify accordingly
])

# === Load trained weights for the MLP mapper ===
model.mapper.load_state_dict(torch.load("weights/vit-b-32-mapper_trained_25epochs.pth", map_location=device))

# === Generate caption using beam search and transformer decoding ===
@torch.no_grad()
def generate_caption(model, image, tokenizer, device, max_length=40, min_length=10, num_beams=5):
    model.eval()
    image = image_transform(image).unsqueeze(0).to(device)

    image_embedding = model.clip_model.encode_image(image)
    prefix_embed = model(image_embedding)

    attention_mask = torch.ones(prefix_embed.shape[:2], dtype=torch.long, device=device)

    outputs = model.gpt2.generate(
        inputs_embeds=prefix_embed,
        attention_mask=attention_mask,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        pad_token_id=tokenizer.eos_token_id
    )

    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return caption if caption.strip() else "No caption generated."

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
    "test_images/classroom.jpg"
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
    caption = generate_caption(model, image, tokenizer, device)
    end_time = time.time()

    inference_time = end_time - start_time
    cider = compute_cider(caption, refs)
    meteor = compute_meteor(caption, refs)
    bleu = compute_bleu(caption, refs)

    display(image)
    print(f"\n===== Image {i+1} =====")
    print(f"Generated Caption: {caption}")
    print(f"Reference Captions: {refs}")
    print(f"Inference Time: {inference_time:.4f} seconds")
    print(f"CIDEr Score:  {cider:.4f}")
    print(f"METEOR Score: {meteor:.4f}")
    print(f"BLEU Score:   {bleu:.4f}")
