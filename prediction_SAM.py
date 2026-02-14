import os
import time
import torch
import clip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms as T
from segment_anything import sam_model_registry

# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 768
CHECKPOINT_PATH = "training_sam_tversky/best_model.pth"
SAM_CHECKPOINT = "sam_vit_b.pth"

TEST_ROOT = "Data/test_labelled"
SAVE_DIR = "predictions/sam"

SAVE_IMAGES_DIR = os.path.join(SAVE_DIR, "images")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_IMAGES_DIR, exist_ok=True)

# ================= MODEL =================
class TextSAM(torch.nn.Module):
    def __init__(self, sam_checkpoint, device):
        super().__init__()

        self.sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)

        if IMG_SIZE != 1024:
            new_size = IMG_SIZE // 16
            pos_embed = self.sam.image_encoder.pos_embed
            pos_embed = pos_embed.permute(0, 3, 1, 2)
            pos_embed = F.interpolate(
                pos_embed,
                size=(new_size, new_size),
                mode="bilinear",
                align_corners=False,
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1)
            self.sam.image_encoder.pos_embed = torch.nn.Parameter(pos_embed)

        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.text_proj = torch.nn.Linear(512, 256)

        self.seg_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(0.3),
            torch.nn.Conv2d(256, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(0.2),
            torch.nn.Conv2d(128, 1, 1)
        )

    def forward(self, image, text):
        image_embed = self.sam.image_encoder(image)

        text_tokens = clip.tokenize([text]).to(DEVICE)
        text_embed = self.clip_model.encode_text(text_tokens).float()
        text_embed = self.text_proj(text_embed)
        text_embed = text_embed.unsqueeze(-1).unsqueeze(-1)

        fused = image_embed + text_embed
        mask = self.seg_head(fused)

        mask = F.interpolate(
            mask,
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False
        )
        return mask

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
model = TextSAM(SAM_CHECKPOINT, DEVICE).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])

# ================= INFERENCE =================
images_dir = os.path.join(TEST_ROOT, "images")
masks_dir = os.path.join(TEST_ROOT, "masks")

results = []
total_time = 0
num_images = 0

for img_name in tqdm(os.listdir(images_dir)):

    img_path = os.path.join(images_dir, img_name)
    image = Image.open(img_path).convert("RGB")

    # Decide prompt from filename
    lower = img_name.lower()
    if "crack" in lower:
        prompt = "segment crack"
    elif "taping" in lower or "joint" in lower:
        prompt = "segment taping area"
    else:
        continue

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    start = time.time()
    with torch.no_grad():
        output = model(image_tensor, prompt)
        pred = torch.sigmoid(output)
    end = time.time()

    total_time += (end - start)
    num_images += 1

    pred_np = pred.squeeze().cpu().numpy()

    # Resize prediction back to ORIGINAL image size
    pred_resized = Image.fromarray((pred_np * 255).astype(np.uint8))
    pred_resized = pred_resized.resize(image.size, Image.NEAREST)
    pred_bin = (np.array(pred_resized) > 127).astype(np.uint8) * 255

    # Save PNG mask
    base_id = os.path.splitext(img_name)[0]
    save_name = f"{base_id}__{prompt.replace(' ', '_')}.png"
    Image.fromarray(pred_bin).save(os.path.join(SAVE_IMAGES_DIR, save_name))

    # -------- Load GT correctly --------
    gt_path = os.path.join(masks_dir, base_id + ".png")
    if not os.path.exists(gt_path):
        continue

    gt = Image.open(gt_path).convert("L")
    gt = gt.resize(image.size, Image.NEAREST)
    gt_np = (np.array(gt) > 127).astype(np.uint8)

    pred_eval = (pred_bin > 127).astype(np.uint8)

    iou, dice = compute_metrics(pred_eval, gt_np)
    results.append([img_name, prompt, iou, dice])

# ================= METRIC REPORT =================
df = pd.DataFrame(results, columns=["image", "prompt", "IoU", "Dice"])
df.to_csv(os.path.join(SAVE_DIR,"test_metrics.csv"), index=False)

print("Mean IoU:", df["IoU"].mean())
print("Mean Dice:", df["Dice"].mean())
print("Avg inference time per image:", total_time / num_images)

# ================= GRAPHS =================
plt.figure()
for prompt in df["prompt"].unique():
    subset = df[df["prompt"] == prompt]
    plt.plot(subset["IoU"].values, label=f"{prompt} IoU")
plt.legend()
plt.title("IoU per Image")
plt.savefig(os.path.join(SAVE_DIR, "IOU_plot.png"))
plt.close()

plt.figure()
for prompt in df["prompt"].unique():
    subset = df[df["prompt"] == prompt]
    plt.plot(subset["Dice"].values, label=f"{prompt} Dice")
plt.legend()
plt.title("Dice per Image")
plt.savefig(os.path.join(SAVE_DIR, "dice_plot.png"))
plt.close()
