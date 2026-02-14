# Prompted Segmentation for Drywall QA

Text-conditioned binary segmentation for drywall defect analysis using:

* **DeepLabV3-ResNet50 + CLIP**
* **Segment Anything Model (ViT-B) + CLIP**

The system generates binary masks (0 / 255) from an input image and a natural language prompt:

* `"segment crack"`
* `"segment taping area"`

## 📌 Project Overview

This project trains and evaluates a **single prompt-conditioned segmentation framework** that:

* Accepts an image + text prompt
* Produces a single-channel PNG mask
* Computes IoU and Dice metrics
* Logs inference time and performance

Two architectures were implemented and compared:

1. DeepLab-based prompt fusion model
2. Text-conditioned SAM-based model

# 📂 Repository Structure

```
Prompted-Segmentation-for-Drywall-QA/
│
├── data/
│   ├── crack/
│   ├── tapping/
│   ├── train_labelled/
│   ├── val_labelled/
│   └── test_labelled/
│
├── predictions/
│
├── train_adam_boundary/
├── training_sam/
│
├── predictions_deeplab.py
├── prediction_SAM.py
└── other utility scripts

```

## 📁 train_labelled / val_labelled / test_labelled

These are the **combined and processed datasets** used for training and evaluation.

After mask generation:

* Crack and taping datasets were merged
* Original 70% / 15% / 15% splits were preserved
* Images and masks are stored together in unified folders

Structure:

```
train_labelled/
 ├── images/
 └── masks/

val_labelled/
 ├── images/
 └── masks/

test_labelled/
 ├── images/
 └── masks/
```

Each mask:

* Single-channel PNG
* Binary values {0,255}
* Filename format: `imageid__prompt.png`

---

## 📁 predictions/

Contains inference outputs from both models.

Includes:

* Saved prediction masks
* IoU & Dice plots
* CSV metric logs

Separate predictions were generated for:

* DeepLab model
* SAM model

---

## 📁 train_adam_boundary/

Contains:

* DeepLab training scripts
* Training logs (CSV)
* Saved checkpoints
* IoU/Dice plots
* Best model weights

Uses:

* AdamW
* Focal + Dice + Boundary loss

---

## 📁 training_sam/

Contains:

* SAM training scripts
* Logs
* Checkpoints
* Metric plots

Uses:

* SAM ViT-B backbone
* CLIP text projection
* Decoder fine-tuning

---

## 🔧 Key Scripts

### `predictions_deeplab.py`

* Loads trained DeepLab + CLIP model
* Performs inference on test set
* Computes IoU & Dice
* Measures average inference time
* Saves PNG masks

### `prediction_SAM.py`

* Loads trained SAM-based model
* Performs prompt-conditioned inference
* Saves predictions
* Computes metrics
* Generates performance plots

---

# 📊 Models Implemented

## 1️⃣ DeepLab + CLIP

* Backbone: DeepLabV3-ResNet50
* Text Encoder: CLIP ViT-B/32
* Fusion: Feature map + projected text embedding
* Faster inference
* Lightweight compared to SAM

---

## 2️⃣ SAM + CLIP

* Backbone: SAM ViT-B
* Text projected into SAM token space
* Mask decoder fine-tuned
* Higher computational cost
* Better structural consistency

---

# 📈 Evaluation

Metrics:

* Mean IoU
* Mean Dice
* Per-image metrics
* Inference time per image

Output:

* PNG masks (0/255)
* CSV logs
* IoU & Dice plots

---

# 🚀 How to Run Inference

### DeepLab

```
python predictions_deeplab.py
```

### SAM

```
python prediction_SAM.py
```

### Data source
`Dataset 1 (Taping area):`
https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect

`Dataset 2 (Cracks):` 
https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36
