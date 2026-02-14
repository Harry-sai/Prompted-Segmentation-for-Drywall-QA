import os
import torch
import clip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
from torchvision.ops import sigmoid_focal_loss
from torchvision.transforms import InterpolationMode
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights
)

# =====================================================
# DEVICE
# =====================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()
print("GPUs available:", NUM_GPUS)

# =====================================================
# CONFIG
# =====================================================

CONFIG = {
    "img_size": 768,
    "batch_size": 8,
    "epochs": 30,
    "lr": 1e-4,
    "weight_decay": 5e-3,
    "num_workers": 4,
    "drop_out":0.3,
    "early_stop_patience":4,
    "comment":"changed optimizer to adamw and loss to sigmod focal , higher epoch and bounday loss"
}


TRAIN_DIR = "Data/train_labelled/"
VAL_DIR = "Data/valid_labelled/"
OUT_DIR = "training_adam_boundary"

os.makedirs(OUT_DIR, exist_ok=True)
pd.DataFrame(CONFIG, index=[0]).to_csv(
    os.path.join(OUT_DIR, "config.csv"), index=False
)


# =====================================================
# DATASET
# =====================================================

class PromptSegDataset(Dataset):
    def __init__(self, root):
        self.image_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.files = sorted(os.listdir(self.image_dir))

        self.transform_img = T.Compose([
            T.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
        ])

        self.transform_mask = T.Compose([
            T.Resize((CONFIG["img_size"], CONFIG["img_size"]),
                     interpolation=InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
    
        img = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")
        mask = Image.open(
            os.path.join(self.mask_dir, fname.replace(".jpg", ".png"))
        )
    
        # Resize first
        img = T.Resize((CONFIG["img_size"], CONFIG["img_size"]))(img)
        mask = T.Resize((CONFIG["img_size"], CONFIG["img_size"]),
                        interpolation=InterpolationMode.NEAREST)(mask)
    
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            img = T.functional.hflip(img)
            mask = T.functional.hflip(mask)
    
        # Random vertical flip
        if torch.rand(1) > 0.5:
            img = T.functional.vflip(img)
            mask = T.functional.vflip(mask)
    
        img = T.ToTensor()(img)
        mask = T.ToTensor()(mask)
        mask = (mask > 0).float()
    
        prompt_text = fname.split("__")[1].replace(".jpg", "").replace("_", " ")
    
        if "crack" in prompt_text:
            prompt_id = torch.tensor(0)
        else:
            prompt_id = torch.tensor(1)
    
        return img, mask, prompt_id

# =====================================================
# MODEL
# =====================================================

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

        # Cache by prompt ID
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
    
                # store on CPU
                self.prompt_cache[pid] = feat
    
            # move to correct GPU replica
            text_embeddings.append(self.prompt_cache[pid].to(device))
    
        text_features = torch.stack(text_embeddings)
    
        text_features = self.text_proj(text_features)
        text_features = text_features.unsqueeze(-1).unsqueeze(-1)
    
        features = self.model.backbone(x)["out"]
        features = features + text_features
    
        out = self.model.classifier(features)
        out = nn.functional.interpolate(
            out,
            size=(CONFIG["img_size"], CONFIG["img_size"]),
            mode="bilinear",
            align_corners=False
        )
    
        return out


# =====================================================
# METRICS
# =====================================================

def compute_metrics(pred, gt):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    iou = intersection / (union + 1e-6)
    dice = (2 * intersection) / (pred.sum() + gt.sum() + 1e-6)
    return iou.item(), dice.item()

def dice_loss(pred, target):
    pred = torch.sigmoid(pred)

    smooth = 1e-6

    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))

    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice.mean()


def boundary_loss(pred, target):
    pred = torch.sigmoid(pred)

    sobel_x = torch.tensor(
        [[1,0,-1],[2,0,-2],[1,0,-1]],
        dtype=torch.float32,
        device=pred.device
    ).view(1,1,3,3)

    sobel_y = sobel_x.permute(0,1,3,2)

    pred_edge = torch.abs(
        nn.functional.conv2d(pred, sobel_x, padding=1)
    ) + torch.abs(
        nn.functional.conv2d(pred, sobel_y, padding=1)
    )

    target_edge = torch.abs(
        nn.functional.conv2d(target, sobel_x, padding=1)
    ) + torch.abs(
        nn.functional.conv2d(target, sobel_y, padding=1)
    )

    return nn.functional.l1_loss(pred_edge, target_edge)
    
def train():

    train_dataset = PromptSegDataset(TRAIN_DIR)
    val_dataset = PromptSegDataset(VAL_DIR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )

    model = PromptedDeepLab()

    if NUM_GPUS > 1:
        model = nn.DataParallel(model)

    model = model.to(DEVICE)

    # ===============================
    # FREEZE BACKBONE INITIALLY
    # ===============================

    backbone = model.module.model.backbone if isinstance(model, nn.DataParallel) else model.model.backbone

    for p in backbone.parameters():
        p.requires_grad = False

    print("Backbone frozen for first 3 epochs.")

    pos_weight = torch.tensor([5.0]).to(DEVICE)
    #bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5
    )

    best_val_dice = 0
    log_data = []

    def evaluate(loader):
        crack_iou, crack_dice = [], []
        tap_iou, tap_dice = [], []

        model.eval()
        with torch.no_grad():
            for imgs, masks, prompt_ids in loader:
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)
                prompt_ids = prompt_ids.to(DEVICE)

                preds = model(imgs, prompt_ids)

                for i in range(len(prompt_ids)):
                    iou, dice = compute_metrics(preds[i], masks[i])
                    if prompt_ids[i].item() == 0:
                        crack_iou.append(iou)
                        crack_dice.append(dice)
                    else:
                        tap_iou.append(iou)
                        tap_dice.append(dice)

        return (np.mean(crack_iou), np.mean(crack_dice),
                np.mean(tap_iou), np.mean(tap_dice))

    for epoch in range(CONFIG["epochs"]):

        # ---- UNFREEZE AFTER 3 EPOCHS ----
        if epoch == 3:
            for p in backbone.parameters():
                p.requires_grad = True
            print("Backbone unfrozen.")

        model.train()
        train_loss = 0

        for imgs, masks, prompt_ids in tqdm(train_loader):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            prompt_ids = prompt_ids.to(DEVICE)

            preds = model(imgs, prompt_ids)
            focal = sigmoid_focal_loss(
                preds, masks,
                alpha=0.75,
                gamma=2,
                reduction="mean"
            )
            
            loss = focal + dice_loss(preds, masks) + 0.1*boundary_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()  


        train_ci, train_cd, train_ti, train_td = evaluate(train_loader)
        val_ci, val_cd, val_ti, val_td = evaluate(val_loader)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Crack IoU:, {train_ci:.4f}")
        print(f"Val Crack IoU:, {val_ci:.4f}")
        print(f"Train Crack Dice:, {train_cd:.4f}")
        print(f"Val Crack Dice:, {val_cd:.4f}")
        print(f"Train Tap IoU:, {train_ti:.4f}")
        print(f"Val Tap IoU:, {val_ti:.4f}")
        print(f"Train Tap Dice:, {train_td:.4f}")
        print(f"Val Tap Dice:, {val_td:.4f}")

        val_mean_dice = (val_cd + val_td) / 2
        
        if val_mean_dice > best_val_dice:
            best_val_dice = val_mean_dice
            early_stop_counter = 0
        
            torch.save(model.state_dict(),
                       os.path.join(OUT_DIR, "best_model.pth"))
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{CONFIG['early_stop_patience']}")
        
            if early_stop_counter >= CONFIG["early_stop_patience"]:
                print("Early stopping triggered.")
                break

        log_data.append([
            epoch+1,
            train_loss/len(train_loader),
            train_ci, train_cd,
            train_ti, train_td,
            val_ci, val_cd,
            val_ti, val_td
        ])

    df = pd.DataFrame(log_data, columns=[
        "epoch","train_loss",
        "train_crack_iou","train_crack_dice",
        "train_tap_iou","train_tap_dice",
        "val_crack_iou","val_crack_dice",
        "val_tap_iou","val_tap_dice"
    ])

    df.to_csv(os.path.join(OUT_DIR, "training_log.csv"), index=False)
    plot_metrics(df)



# =====================================================
# PLOTTING
# =====================================================

def plot_metrics(df):

    fig, axs = plt.subplots(1, 2, figsize=(12,5))

    axs[0].plot(df["epoch"], df["train_crack_iou"], label="Train")
    axs[0].plot(df["epoch"], df["val_crack_iou"], label="Val")
    axs[0].set_title("Crack IoU")
    axs[0].legend()

    axs[1].plot(df["epoch"], df["train_tap_iou"], label="Train")
    axs[1].plot(df["epoch"], df["val_tap_iou"], label="Val")
    axs[1].set_title("Taping IoU")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "iou_subplot.png"))
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(12,5))

    axs[0].plot(df["epoch"], df["train_crack_dice"], label="Train")
    axs[0].plot(df["epoch"], df["val_crack_dice"], label="Val")
    axs[0].set_title("Crack Dice")
    axs[0].legend()

    axs[1].plot(df["epoch"], df["train_tap_dice"], label="Train")
    axs[1].plot(df["epoch"], df["val_tap_dice"], label="Val")
    axs[1].set_title("Taping Dice")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "dice_subplot.png"))
    plt.close()

# =====================================================

if __name__ == "__main__":
    train()
