#!/usr/bin/env python3
"""Train a binary 2D U-Net while preserving the existing Cellpose preparation.

This script deliberately retains these project conventions:
- ``access.txt`` with ND2 path in ``content[1]`` and annotation root in ``content[2]``;
- raw images as ``data[recording_name][frame_index]``;
- masks as ``annotation[recording_name][frame_index]``;
- frame splitting with ``Trainingpreperation().split_frames_by_key(frameindex)``;
- training-only augmentation with ``spatial_augmentation`` and ``pixelaugmentation``.

The only replacement is the final Cellpose training stage: it trains/saves a
PyTorch U-Net (binary foreground/background segmentation).
"""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import random
import re
from pathlib import Path

import nd2
import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from training import Trainingpreperation
from anhs_watershed import Background


# ========================== Configuration =======================================
NUM_AUGMENTATIONS_PER_FRAME = 2
EPOCHS = 300
BATCH_SIZE = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BASE_CHANNELS = 32
SEED = 42
OUTPUT = Path.cwd() / "cellpose_training"  # Existing annotation TIFF location
MODEL_DIR = Path.cwd() / "unet_models"


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =================== Existing data preparation (kept) ===========================
with open("access.txt", "r", encoding="utf-8") as file:
    content = [line.strip() for line in file]

# Kept equivalent to: Path(str(Path.cwd()) + content[1]).
# ``lstrip('/\\')`` also works if the access entry starts with a slash.
data_directory = Path.cwd() / content[1].lstrip("/\\")
datapath = list(data_directory.glob("*.nd2"))

data = {}
for file in datapath:
    # Kept for consistency with your original data loading. The model uses the
    # corrected image below, exactly as in the Cellpose code.
    raw = nd2.imread(file).astype(np.float32)
    new_img = Background().darkfield_assignment(file)[0]["raw_minus_dark"]
    data[file.stem] = np.asarray(new_img, dtype=np.float32)

# Collect the original ImageJ annotation files. This is retained in case you need
# to uncomment the conversion block; training below reads the already exported TIFFs.
maskpath = [folder for folder in (Path.cwd() / content[2].lstrip("/\\")).iterdir() if folder.is_dir()]
mask_files = {folder.name: list(folder.glob("*")) for folder in maskpath}

# Convert .zip/.roi files to TIFF only when TIFF masks have not already been made.
# OUTPUT.mkdir(exist_ok=True)
# for name, files in mask_files.items():
#     for p in files:
#         Trainingpreperation().process_rois_to_tiff(str(p), str(OUTPUT), 600, 600, name)

# Annotation dictionary: ``annotation[folder_name][frame_index] = mask array``.
annotation = {}
frameindex = {}

for folder in OUTPUT.iterdir():
    if folder.is_dir():
        listholder = {}
        placeholder = []
        for mask_file in folder.glob("*.tif*"):
            match = re.search(r"(\d+)\.tif{1,2}$", mask_file.name, flags=re.IGNORECASE)
            if match is None:
                print(f"Skipping non-frame TIFF: {mask_file.name}")
                continue
            frame = int(match.group(1))
            mask = np.squeeze(tiff.imread(mask_file))
            if mask.ndim != 2:
                raise ValueError(f"{mask_file} has shape {mask.shape}; expected a 2D mask.")
            listholder[frame] = (mask > 0).astype(np.float32)  # binary U-Net target
            placeholder.append(frame)

        if listholder:
            annotation[folder.name] = listholder
            frameindex[folder.name] = sorted(placeholder)

# Checks before training: avoid silently pairing wrong images/masks.
if not annotation:
    raise RuntimeError(f"No TIFF masks found below {OUTPUT}.")
for key, frames in frameindex.items():
    if key not in data:
        raise KeyError(f"Annotation folder '{key}' has no corresponding ND2 file basename.")
    for frame_idx in frames:
        if frame_idx >= len(data[key]):
            raise IndexError(f"{key}: mask frame {frame_idx} exceeds ND2 frame range (0..{len(data[key]) - 1}).")
        if data[key][frame_idx].shape != annotation[key][frame_idx].shape:
            raise ValueError(
                f"{key}, frame {frame_idx}: ND2 image shape {data[key][frame_idx].shape} "
                f"does not match mask shape {annotation[key][frame_idx].shape}."
            )

# ================= Existing split and augmentation logic (kept) =================
train_frames, val_frames, test_frames = Trainingpreperation().split_frames_by_key(frameindex)

train_images, train_masks = [], []
val_images, val_masks = [], []
test_images, test_masks = [], []
prep = Trainingpreperation()

for key, frames in train_frames.items():
    for frame_idx in frames:
        # Original unaugmented pair
        train_images.append(data[key][frame_idx])
        train_masks.append(annotation[key][frame_idx])

        # Original project augmentation functions, only for training data
        for _ in range(NUM_AUGMENTATIONS_PER_FRAME):
            aug_img, aug_mask = prep.spatial_augmentation(data[key][frame_idx], annotation[key][frame_idx])
            aug_img = prep.pixelaugmentation(aug_img)
            train_images.append(aug_img)
            train_masks.append(aug_mask)

for key, frames in val_frames.items():
    for frame_idx in frames:
        val_images.append(data[key][frame_idx])
        val_masks.append(annotation[key][frame_idx])

for key, frames in test_frames.items():
    for frame_idx in frames:
        test_images.append(data[key][frame_idx])
        test_masks.append(annotation[key][frame_idx])

print(f"Total training samples: {len(train_images)}")
print(f"Total validation samples: {len(val_images)}")
print(f"Total test samples: {len(test_images)}")

if not train_images or not val_images:
    raise RuntimeError("Training or validation set is empty. Check split_frames_by_key and annotation count.")


# ============================= Device selection =================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"NVIDIA GPU (CUDA) is active: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Apple Silicon GPU (MPS) is active.")
else:
    device = torch.device("cpu")
    print("GPU not available; defaulting to CPU.")


# =============================== U-Net ==========================================
class SegmentationDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.asarray(self.images[index], dtype=np.float32)
        mask = (np.asarray(self.masks[index]) > 0).astype(np.float32)

        # Per-frame robust scaling makes darkfield-corrected intensities stable.
        low, high = np.percentile(image, (1, 99))
        image = np.zeros_like(image) if high <= low else np.clip((image - low) / (high - low), 0, 1)
        return torch.from_numpy(np.ascontiguousarray(image[None])), torch.from_numpy(np.ascontiguousarray(mask[None]))


class DoubleConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out), nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, base=BASE_CHANNELS):
        super().__init__()
        self.e1, self.e2 = DoubleConv(1, base), DoubleConv(base, base * 2)
        self.e3, self.e4 = DoubleConv(base * 2, base * 4), DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.b = DoubleConv(base * 8, base * 16)
        self.u4, self.d4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2), DoubleConv(base * 16, base * 8)
        self.u3, self.d3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2), DoubleConv(base * 8, base * 4)
        self.u2, self.d2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2), DoubleConv(base * 4, base * 2)
        self.u1, self.d1 = nn.ConvTranspose2d(base * 2, base, 2, 2), DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    @staticmethod
    def resize(x, reference):
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False) if x.shape[-2:] != reference.shape[-2:] else x

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))
        b = self.b(self.pool(e4))
        d4 = self.d4(torch.cat([e4, self.resize(self.u4(b), e4)], dim=1))
        d3 = self.d3(torch.cat([e3, self.resize(self.u3(d4), e3)], dim=1))
        d2 = self.d2(torch.cat([e2, self.resize(self.u2(d3), e2)], dim=1))
        d1 = self.d1(torch.cat([e1, self.resize(self.u1(d2), e1)], dim=1))
        return self.out(d1)  # logits, no sigmoid here


def dice_loss(logits, target, eps=1e-6):
    prediction = torch.sigmoid(logits)
    intersection = (prediction * target).sum((1, 2, 3))
    total = prediction.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return (1 - (2 * intersection + eps) / (total + eps)).mean()


def loss_function(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    losses, dices, count = 0.0, 0.0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        batch_size = images.size(0)
        losses += loss_function(logits, masks).item() * batch_size
        pred = (torch.sigmoid(logits) >= 0.5).float()
        inter = (pred * masks).sum((1, 2, 3))
        denom = pred.sum((1, 2, 3)) + masks.sum((1, 2, 3))
        dices += ((2 * inter + 1e-6) / (denom + 1e-6)).sum().item()
        count += batch_size
    return losses / count, dices / count


# =============================== Training =======================================
seed_everything()
pin_memory = device.type == "cuda"
train_loader = DataLoader(SegmentationDataset(train_images, train_masks), batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_memory)
val_loader = DataLoader(SegmentationDataset(val_images, val_masks), batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory)
test_loader = DataLoader(SegmentationDataset(test_images, test_masks), batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory) if test_images else None

model = UNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=12)
MODEL_DIR.mkdir(exist_ok=True)
best_model_path = MODEL_DIR / "unet_best.pth"
last_model_path = MODEL_DIR / "unet_last.pth"
best_val_dice = -1.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, items = 0.0, 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(images), masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        items += images.size(0)

    train_loss = running_loss / items
    val_loss, val_dice = evaluate(model, val_loader)
    scheduler.step(val_dice)
    checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "val_dice": val_dice,
                  "base_channels": BASE_CHANNELS, "threshold": 0.5}
    torch.save(checkpoint, last_model_path)
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(checkpoint, best_model_path)
    print(f"Epoch {epoch:03d}/{EPOCHS} | train loss={train_loss:.4f} | val loss={val_loss:.4f} | val Dice={val_dice:.4f}")

# Always report test performance from the best validation checkpoint.
model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=False)["model_state_dict"])
if test_loader:
    test_loss, test_dice = evaluate(model, test_loader)
    print(f"Final test loss={test_loss:.4f} | Final test Dice={test_dice:.4f}")
print(f"Best U-Net checkpoint saved to: {best_model_path}")
