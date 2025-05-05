# 🖼️ EdgeCap: Real-Time Image Captioning at the Edge

**Lightweight Image Captioning Vision-Language Model Deployment on Raspberry Pi 4B**

**EdgeCap** is a lightweight vision-language model pipeline that runs directly on a Raspberry Pi 4B (8GB RAM, Ubuntu) using a connected camera module. It combines **TinyCLIP** for image understanding with **DistilGPT2** for caption generation, leveraging PyTorch to enable efficient, real-time or near-real-time image captioning on low-power edge devices. The system also supports the full **CLIP ViT-B/32** variant for higher accuracy when needed.

---

## 🚀 Key Features

- ✅ Runs on Raspberry Pi 4B with Ubuntu
- 🧠 Vision-Language architecture: CLIP or TinyCLIP + Distilled GPT-2
- 🔍 Supports both **full-size** and **lightweight (TinyCLIP)** CLIP backbones
- 📸 Works with the Raspberry Pi camera module
- ⚡ Real-time inference possible with TinyCLIP on edge hardware

---

## 📦 Model Variants

| Variant      | Visual Encoder         | Text Decoder   | Description                              |
|--------------|------------------------|----------------|------------------------------------------|
| **EdgeCap-16** | TinyCLIP-ViT-39M-16 (wkcn)    | DistilGPT2     | Fast, low-memory version for edge devices |
| **EdgeCap-32** | CLIP ViT-B/32 (OpenCLIP) | DistilGPT2     | Larger, more accurate variant             |

---

## 🧪 Evaluation

Evaluated on [Flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k) using:

* **BLEU**
* **CIDEr**
* **METEOR**

---

## 🤖 Deployment Details

- **Device:** Raspberry Pi 4B (8GB RAM)
- **OS:** Ubuntu 22.04
- **Libraries:** PyTorch, OpenCLIP, HuggingFace Transformers
- **Camera Support:** USB / Pi camera module
- **Input Options:** Live camera feed or local image files

---

## 📂 Project Structure

```bash
edgecap/
├── model/
│   ├── tinyclip_gpt2.py        # MLP mapper + TinyCLIP + distilgpt2
│   └── fullclip_gpt2.py        # ViT-B/32 + distilgpt2
├── scripts/
│   ├── run_inference.py        # Generate captions
│   ├── evaluate_metrics.py     # BLEU, METEOR, CIDEr
│   └── capture_camera.py       # Capture via Pi camera
├── data/
│   └── flickr_subset.json
└── README.md
```

---

## 📈 Example Output

| Image                   | Caption                              |
| ----------------------- | ------------------------------------ |
| ![soccer](example1.jpg) | "Soccer players on a field"          |
| ![dog](example2.jpg)    | "A golden retriever chewing a treat" |

---

## 💡 Future Enhancements

* Quantization & pruning for faster edge inference
* Support for real-time camera streaming
* Model distillation and further lightweighting

---

## 🙏 Acknowledgements

This project leverages the following open-source contributions:

* [OpenCLIP](https://github.com/mlfoundations/open_clip) for CLIP and TinyCLIP models
* [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) for `distilgpt2`
* [PyTorch](https://pytorch.org/) for model implementation and deployment
* [Flickr30k Dataset](https://huggingface.co/datasets/nlphuji/flickr30k) for evaluation
* [pycocoevalcap](https://github.com/tylin/coco-caption) for evaluation metrics

We thank the maintainers of these libraries and datasets for their invaluable work.
