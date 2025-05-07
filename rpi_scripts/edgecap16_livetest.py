import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from PIL import Image
from torchvision import transforms
import open_clip

from tqdm import tqdm
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import nltk

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

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
device="cpu"
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

# Load CLIP model
clip_model = TinyCLIPCaptioner(device="cpu")

# Load GPT2
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
gpt2 = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
gpt2.eval()

# Assemble TinyCLIPCap
model = TinyCLIPCap(clip_model, gpt2).to(device)

"""**train mlp mapper**"""

from torchvision import transforms
from PIL import Image
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    ),
])



image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # if using a single-channel image; adjust for RGB
])

# Load trained mapper weights (if available)
model.mapper.load_state_dict(torch.load("mapper_trained_epoch24.pth", map_location=device))

def generate_caption(model, image, tokenizer, device, max_length=30):
    model.eval()
    with torch.no_grad():
        # Transform and move image to device
        image = image_transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

        # Get image embedding and prefix
        image_embedding = model.clip_model.encode_image(image)
        prefix_embed = model(image_embedding)  # [1, prefix_len, 768]

        # Start token
        generated = torch.tensor(tokenizer.encode(tokenizer.bos_token), device=device).unsqueeze(0)

        for _ in range(max_length):
            caption_embed = model.gpt2.transformer.wte(generated)
            gpt2_input = torch.cat((prefix_embed, caption_embed), dim=1)
            attention_mask = torch.ones(gpt2_input.shape[:2], dtype=torch.long, device=device)

            outputs = model.gpt2(inputs_embeds=gpt2_input, attention_mask=attention_mask)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            generated = torch.cat((generated, next_token), dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

        caption = tokenizer.decode(generated.squeeze(), skip_special_tokens=True)
        return caption

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters: {count_trainable_params(model):,}")

print(f"Trainable parameters in mapper: {count_trainable_params(model.mapper):,}")




@torch.no_grad()
def generate_caption(model, image, tokenizer, device, max_length=40, min_length=10, num_beams=5):
    model.eval()
    image = image_transform(image).unsqueeze(0).to(device)

    image_embedding = model.clip_model.encode_image(image)
    prefix_embed = model(image_embedding)

    # Create attention mask: 1s for prefix
    attention_mask = torch.ones(prefix_embed.shape[:2], dtype=torch.long, device=device)

    outputs = model.gpt2.generate(
        inputs_embeds=prefix_embed,
        attention_mask=attention_mask,    # ? fix warning
        max_length=max_length,             # Maximum words
        min_length=min_length,             # ? Force at least this many words
        num_beams=num_beams,                # ? Beam search (try multiple paths)
        early_stopping=True,                # Stop when all beams finished
        no_repeat_ngram_size=3,             # ? Avoid repeating same 3-word sequences
        length_penalty=1.0,                 # 1.0 = balanced; >1 favors longer, <1 favors shorter
        pad_token_id=tokenizer.eos_token_id
    )

    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if caption.strip() == "":
        caption = "No caption generated."
    return caption

# Dummy image tensor (matches expected shape)
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# Run a dummy forward pass through the pipeline to trigger weight loading & compilation
with torch.no_grad():
    dummy_embed = model.clip_model.encode_image(dummy_input)
    dummy_prefix = model(dummy_embed)
    dummy_gpt_input = torch.cat(
        (dummy_prefix, model.gpt2.transformer.wte(torch.tensor([[tokenizer.bos_token_id]], device=device))),
        dim=1
    )
    _ = model.gpt2(inputs_embeds=dummy_gpt_input)


import requests
from io import BytesIO
import time
from PIL import Image


image = Image.open("/home/ubuntu/livingroom.jpg").convert("RGB")

start_time = time.time()
caption = generate_caption(model, image, tokenizer, device)
end_time = time.time()

inference_time = end_time - start_time

print("Generated Caption from Pi Camera:", caption)
print(f"Inference Time: {inference_time:.4f} seconds")