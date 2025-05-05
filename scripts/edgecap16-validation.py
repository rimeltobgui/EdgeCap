# === Import required libraries ===
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from PIL import Image

import time
from torchvision import transforms
import open_clip

from datasets import load_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

import nltk
nltk.download('punkt')
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
            clip_dim=512,
            gpt2_dim=gpt2_model.config.n_embd,
            prefix_length=prefix_length
        )

    def forward(self, image_embedding):
        return self.mapper(image_embedding)

# === Set device for inference ===
device = torch.device("cuda")

# === Load TinyCLIP-ViT-39M-16 model ===
class TinyCLIPCaptioner:

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

    def encode_image(self, image):
        return self.model.encode_image(image)

clip_model = TinyCLIPCaptioner(device=device)

# === Load DistilGPT2 and tokenizer ===
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
gpt2 = GPT2LMHeadModel.from_pretrained("distilgpt2")
gpt2.eval()
gpt2 = gpt2.to(device)

# === Assemble full model ===
model = TinyCLIPCap(clip_model, gpt2).to(device)

# === Setup tokenizer pad token ===
tokenizer.pad_token = tokenizer.eos_token

# === Normalize input images ===
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # for RGB inputs, modify accordingly
])

# === Load trained weights for the MLP mapper ===
model.mapper.load_state_dict(torch.load("weights/edgecap16-mapper_trained_25epochs.pth", map_location=device))

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


# === Load Flicker30k Dataset ===

dataset = load_dataset("nlphuji/flickr30k", split="test[:90%]")  #
print(f"Loaded {len(dataset)} samples")

# === PyTorch-compatible dataset wrapper for Flickr30k ===

class FlickrDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, transform):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = item["image"]
        image = self.transform(img)

        # 👇 pick just one caption randomly
        caption = item["caption"]
        if isinstance(caption, list):
            caption = caption[0]  # or: random.choice(caption)

        tokenized = self.tokenizer(
            caption,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=30,
        )
        input_ids = tokenized["input_ids"].squeeze(0)         # [seq_len]
        attention_mask = tokenized["attention_mask"].squeeze(0)  # [seq_len]

        return image, input_ids, attention_mask

# === Load data loader ===

train_dataset = FlickrDataset(dataset, tokenizer, image_transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


# === Caption generation loop for a batch of images ===

def generate_captions(model, dataloader, tokenizer, device):
    model.eval()
    all_results = []

    for images, _, _ in tqdm(dataloader):
        images = images.to(device)

        # Get image embeddings
        with torch.no_grad():
            image_embeddings = model.clip_model.encode_image(images)
            prefix_embed = model(image_embeddings)  # [B, prefix_len, 768]

            # Prepare input: start with prefix only
            generated_ids = model.gpt2.generate(
                inputs_embeds=prefix_embed,
                max_length=40,
                num_beams=5,
                do_sample=False,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
            )

            captions = [tokenizer.decode(g, skip_special_tokens=True).strip() for g in generated_ids]

        # Move images to CPU for display
        images_cpu = images.cpu()

        for img_tensor, caption in zip(images_cpu, captions):
            img_pil = TF.to_pil_image(img_tensor)
            all_results.append((img_pil, caption))

    return all_results

# === Run inference and store results ===
results = generate_captions(model, train_loader, tokenizer, device)


# === Visualize first 10 captioned images ===
for idx, (image, caption) in enumerate(results[:10]):  # show first 10
    plt.imshow(image)
    plt.title(caption)
    plt.axis("off")
    plt.show()

# === Metric scoring functions ===
cc = SmoothingFunction()
meteor_scorer = Meteor()

def compute_bleu_scores(predictions, references):
    scores = []
    for pred, refs in zip(predictions, references):
        ref_tokenized = [r.lower().split() for r in refs]
        pred_tokenized = pred.lower().split()
        score = sentence_bleu(ref_tokenized, pred_tokenized, smoothing_function=cc.method1)
        scores.append(score)
    return sum(scores) / len(scores)

def compute_cider_scores(predictions, references):
    gts = {str(i): [{'image_id': str(i), 'caption': r} for r in refs] for i, refs in enumerate(references)}
    res = {str(i): [{'image_id': str(i), 'caption': pred}] for i, pred in enumerate(predictions)}
    tokenizer = PTBTokenizer()
    gts_tokenized = tokenizer.tokenize(gts)
    res_tokenized = tokenizer.tokenize(res)
    scorer = Cider()
    score, _ = scorer.compute_score(gts_tokenized, res_tokenized)
    return score

def compute_meteor_scores(predictions, references):
    scorer = Meteor()
    gts = {str(i): refs for i, refs in enumerate(references)}
    res = {str(i): [pred] for i, pred in enumerate(predictions)}
    score, _ = scorer.compute_score(gts, res)
    return score

# === Master evaluation wrapper ===
def evaluate_captions(predictions, references):
    return {
        "BLEU": compute_bleu_scores(predictions, references),
        "CIDEr": compute_cider_scores(predictions, references),
        "METEOR": compute_meteor_scores(predictions, references),
    }

# === Caption and evaluate Flicker30k Dataset ===
# 1. Extract generated captions
generated_captions = [cap for (_, cap) in results]

# 2. Extract ground truth captions
reference_captions = []
for i in range(len(results)):
    refs = train_dataset.data[i]["caption"]
    if isinstance(refs, str):
        refs = [refs]
    reference_captions.append(refs)

# 3. Evaluate
metrics = evaluate_captions(generated_captions, reference_captions)

# 4. Print results
print("\n===== Evaluation Metrics =====")
for metric_name, score in metrics.items():
    print(f"{metric_name}: {score:.4f}")
