# 🖼️ EdgeCap: Real-Time Image Captioning at the Edge

**Lightweight Image Captioning Vision-Language Model Deployment on Raspberry Pi 4B**

**EdgeCap** is a lightweight vision-language model pipeline that runs directly on a Raspberry Pi 4B (8GB RAM, Ubuntu) using a connected camera module. It combines **TinyCLIP** for image understanding with **DistilGPT2** for caption generation, leveraging PyTorch to enable efficient, real-time or near-real-time image captioning on low-power edge devices. The system also supports the full **CLIP ViT-B/32** variant for higher accuracy when needed.

---

## 🚀 Key Features

- Runs on Raspberry Pi 4B with Ubuntu
- Vision-Language architecture: CLIP or TinyCLIP + Distilled GPT-2
- Supports both **full-size** and **lightweight (TinyCLIP)** CLIP backbones
- Works with the Raspberry Pi camera module
- Real-time inference possible with TinyCLIP on edge hardware

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

## 🔧 Project Structure

```markdown
📦 EdgeCap
├── scripts/                # Inference, training, and validation scripts
│   ├── edgecap16-*.py      # TinyCLIP (39M) + DistilGPT2 variant
│   ├── edgecap32-*.py      # ViT-B/32 + DistilGPT2 variant
│   ├── tinyclip-*.py       # TinyCLIP (39M) inference
│   └── blip-*.py           # BLIP baseline and quantization scripts
├── test_images/            # Example images for evaluation
├── weights/                # Pretrained mapper weights
├── full_project_notebook.ipynb
├── requirements.txt
└── README.md
```

---

## 📓 Notebook Version

Check out [full_project_notebook.ipynb](full_project_notebook.ipynb) for an interactive demonstration in Google Colab.

---

## 📈 Example Output

| Image                   | Caption                              |
| ----------------------- | ------------------------------------ |
| ![soccer](test_images/soccer.jpg) | "two soccer players in a soccer uniform, one wearing a red shirt and the other wearing a white shirt"          |
| ![dog](test_images/dog.jpg)    | "a golden retriever is holding a toy in his mouth" |

---

## 🛠️ Setup
```bash
# Clone the EdgeCap repository and install dependencies
git clone https://github.com/rimeltobgui/EdgeCap.git && cd EdgeCap
pip install -r requirements.txt
```

---

## 💡 Future Enhancements

* Quantization & pruning for faster edge inference
* Support for real-time camera streaming
* Model distillation and further lightweighting

---

## Authors

* Esraa Fahmy 100062654@ku.ac.ae, 
* Rim ElTobgui 100063155@ku.ac.ae  
Khalifa University
---

## 🙏 Acknowledgements

This project builds upon the open-source work of:

- [TinyCLIP (wkcn)](https://github.com/wkcn/TinyCLIP)
- [OpenCLIP (MLFoundations)](https://github.com/mlfoundations/open_clip)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
* [PyTorch](https://pytorch.org/) for model implementation and deployment
* [Flickr30k Dataset](https://huggingface.co/datasets/nlphuji/flickr30k) for evaluation
* [pycocoevalcap](https://github.com/tylin/coco-caption) for evaluation metrics

We thank the maintainers of these libraries and datasets for their invaluable work.
