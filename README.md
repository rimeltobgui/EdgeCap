# 🖼️ EdgeCap: Real-Time Image Captioning at the Edge Using Lightweight Vision-Language Models

**EdgeCap** is a lightweight vision-language model pipeline that runs directly on a Raspberry Pi 4B (8GB RAM) using a connected camera module. It combines **TinyCLIP** for image understanding with **DistilGPT2** for caption generation, optimized for low-power, edge-device inference using PyTorch on Ubuntu.

---

## 📷 Project Highlights

* **Device**: Raspberry Pi 4B (8GB RAM) running Ubuntu
* **Input**: Live camera feed or static images
* **Model Architecture**:

  * **TinyCLIP** (ViT-32 or TinyCLIP-ViT-39M-16-Text-19M) for visual encoding
  * **DistilGPT2** as a lightweight language model
  * **MLP Mapper** to project visual features into GPT2’s input space
* **Library Stack**: PyTorch, Transformers, OpenCLIP, torchvision
* **Use Case**: Captioning images on the edge with minimal latency

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/edgecap-pi.git
cd edgecap-pi
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure your Raspberry Pi has:

* Python 3.9+
* PyTorch (CPU)
* OpenCLIP
* Transformers
* torchvision
* PIL

### 3. (Optional) Load Pretrained Mapper

```bash
wget https://your-model-link/mapper_trained.pth -O mapper_trained.pth
```

---

## 🎯 Run Inference

```bash
python run_edgecap.py --image path/to/image.jpg
```

Or run on a live camera feed:

```bash
python camera_inference.py
```

---

## 🧠 Model Training (Optional)

The MLP mapper can be trained using Flickr30k. See `train_mapper.py` for training instructions on GPU-enabled machines.

---

## 📊 Evaluation

* BLEU, CIDEr, and METEOR metrics supported
* Evaluate using `evaluate.py` on a subset of Flickr30k

---

## 📁 Directory Structure

```
edgecap/
│
├── models/                # TinyCLIP + GPT2 + Mapper
├── camera_inference.py    # Live captioning from Pi Camera
├── run_edgecap.py         # Inference on static images
├── train_mapper.py        # MLP training script
├── evaluate.py            # Caption quality evaluation
├── requirements.txt
└── README.md
```

---

## ⚙️ Future Plans

* Quantization-aware training for further memory savings
* Integration with Jetson Nano and Coral Edge TPU
* Support for object detection + caption fusion

---

## 🧪 License

MIT License. See `LICENSE`.

---

Would you like me to write the `requirements.txt` file next?

