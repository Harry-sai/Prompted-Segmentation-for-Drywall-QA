import os
import time
import torch
import clip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "img_size": 768,
    "drop_out": 0.3
}

CHECKPOINT_PATH = "training_adam_boundary/best_model.pth"
TEST_ROOT = "Data/test_labelled"
SAVE_DIR = "predictions/deeplab"
SAVE_MASK_DIR = os.path.join(SAVE_DIR, "images")

VALID_EXT = [".jpg", ".jpeg", ".png"]

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_MASK_DIR, exist_ok=True)

# ================= MODEL =================
class PromptedDeepLab(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = deeplabv3_resnet50(
            weights=DeepLabV3_ResNet50_Weights.DEFAULT
        )

        self.model.classifier[4] = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(CONFIG['drop_out']),
            nn.Conv2d(256, 1, 1)
        )

        self.clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.text_proj = nn.Linear(512, 2048)
        self.prompt_cache = {}

    def forward(self, x, prompt_ids):

        device = x.device
        text_embeddings = []

        for pid in prompt_ids:
            pid = pid.item()

            if pid not in self.prompt_cache:

                if pid == 0:
                    text = "segment crack"
                else:
                    text = "segment taping area"

                with torch.no_grad():
                    tokens = clip.tokenize([text]).to(device)
                    feat = self.clip_model.encode_text(tokens).float()[0].cpu()

                self.prompt_cache[pid] = feat

            text_embeddings.append(self.prompt_cache[pid].to(device))

        text_features = torch.stack(text_embeddings)
        text_features = self.text_proj(text_features)
        text_features = text_features.unsqueeze(-1).unsqueeze(-1)

        features = self.model.backbone(x)["out"]
        features = features + text_features

        out = self.model.classifier(features)

        out = F.interpolate(
            out,
            size=(CONFIG["img_size"], CONFIG["img_size"]),
            mode="bilinear",
            align_corners=False
        )

        return out

# ================= METRICS =================
def compute_metrics(pred, gt):
    pred = pred > 0.5
    gt = gt > 0.5

    intersection = (pred & gt).sum()
    union = (pred | gt).sum()

    if union == 0:
        return 1.0, 1.0

    iou = intersection / (union + 1e-6)
    dice = (2 * intersection) / (pred.sum() + gt.sum() + 1e-6)
    return iou, dice

# ================= LOAD MODEL =================
model = PromptedDeepLab().to(DEVICE)
state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)

model.eval()

transform = T.Compose([
    T.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    T.ToTensor()
])

# ================= INFERENCE =================
images_dir = os.path.join(TEST_ROOT, "images")
masks_dir = os.path.join(TEST_ROOT, "masks")

results = []
total_time = 0
num_images = 0

image_files = [
    f for f in os.listdir(images_dir)
    if os.path.splitext(f)[1].lower() in VALID_EXT
]

for img_name in tqdm(image_files):

    base_id = os.path.splitext(img_name)[0]
    img_path = os.path.join(images_dir, img_name)

    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # -------- prompt id --------
    lower = img_name.lower()
    if "crack" in lower:
        prompt_id = 0
    else:
        prompt_id = 1

    prompt_ids = torch.tensor([prompt_id]).to(DEVICE)

    # -------- inference --------
    start = time.time()
    with torch.no_grad():
        output = model(image_tensor, prompt_ids)
        pred = torch.sigmoid(output)
    end = time.time()

    total_time += (end - start)
    num_images += 1

    pred_np = pred.squeeze().cpu().numpy()

    # Resize back to original image size
    pred_resized = Image.fromarray((pred_np * 255).astype(np.uint8))
    pred_resized = pred_resized.resize(image.size, Image.NEAREST)

    pred_bin = (np.array(pred_resized) > 127).astype(np.uint8)

    # -------- save mask --------
    save_name = f"{base_id}__pred.png"
    Image.fromarray(pred_bin * 255).save(
        os.path.join(SAVE_MASK_DIR, save_name)
    )

    # -------- load GT --------
    gt_path = None
    for ext in VALID_EXT:
        possible = os.path.join(masks_dir, base_id + ext)
        if os.path.exists(possible):
            gt_path = possible
            break

    if gt_path is None:
        continue

    gt = Image.open(gt_path).convert("L")
    gt = gt.resize(image.size, Image.NEAREST)
    gt_np = (np.array(gt) > 127).astype(np.uint8)

    # -------- metrics --------
    iou, dice = compute_metrics(pred_bin, gt_np)

    results.append([img_name, prompt_id, iou, dice])

# ================= SAVE RESULTS =================
df = pd.DataFrame(results, columns=["image", "prompt_id", "IoU", "Dice"])
df.to_csv(os.path.join(SAVE_DIR, "test_metrics.csv"), index=False)

print("Mean IoU :", df["IoU"].mean())
print("Mean Dice:", df["Dice"].mean())
print("Avg inference time per image:", total_time / max(1, num_images))

# ================= PLOTS =================
plt.figure()
for pid in df["prompt_id"].unique():
    subset = df[df["prompt_id"] == pid]
    label = "crack" if pid == 0 else "taping"
    plt.plot(subset["IoU"].values, label=f"{label} IoU")
plt.legend()
plt.title("IoU per Image")
plt.savefig(os.path.join(SAVE_DIR, "iou_plot.png"))
plt.close()

plt.figure()
for pid in df["prompt_id"].unique():
    subset = df[df["prompt_id"] == pid]
    label = "crack" if pid == 0 else "taping"
    plt.plot(subset["Dice"].values, label=f"{label} Dice")
plt.legend()
plt.title("Dice per Image")
plt.savefig(os.path.join(SAVE_DIR, "dice_plot.png"))
plt.close()

print("Testing complete.")
