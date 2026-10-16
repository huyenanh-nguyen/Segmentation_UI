#!/usr/bin/env python3
"""Apply a trained U-Net to darkfield-corrected ND2 frames and export masks.

Uses the same project structure as train_unet_nd2_compatible.py:
- access.txt: ND2 folder in content[1]
- Background().darkfield_assignment(): darkframe correction
- unet_models/unet_best.pth: trained checkpoint

Outputs:
  unet_predictions/<ND2_basename>/frame_0000_probability.tif  (float32, 0..1)
  unet_predictions/<ND2_basename>/frame_0000_binary.tif       (0/1 mask)
  unet_predictions/<ND2_basename>/frame_0000_instances.tif    (connected components)
  unet_predictions/annotated_test_metrics.csv                  (if TIFF ground truths exist)
  unet_predictions/qualitative_examples.png                    (if TIFF ground truths exist)
"""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import nd2
import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label

from anhs_watershed import Background
from training import Trainingpreperation

# ========================== Configuration =======================================
CHECKPOINT_PATH = Path.cwd() / "unet_models" / "unet_best.pth"
PREDICTION_DIR = Path.cwd() / "unet_predictions"
ANNOTATION_DIR = Path.cwd() / "cellpose_training"
THRESHOLD = 0.5
BATCH_SIZE = 2
SAVE_PROBABILITY_MAPS = True
SAVE_BINARY_MASKS = True
SAVE_INSTANCE_MASKS = True


# ==================== Same UNet architecture as training =======================
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
    def __init__(self, base=32):
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
        if x.shape[-2:] != reference.shape[-2:]:
            return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))
        b = self.b(self.pool(e4))
        d4 = self.d4(torch.cat([e4, self.resize(self.u4(b), e4)], dim=1))
        d3 = self.d3(torch.cat([e3, self.resize(self.u3(d4), e3)], dim=1))
        d2 = self.d2(torch.cat([e2, self.resize(self.u2(d3), e2)], dim=1))
        d1 = self.d1(torch.cat([e1, self.resize(self.u1(d2), e1)], dim=1))
        return self.out(d1)


# ============================= Utility functions ===============================
def device_for_inference():
    if torch.cuda.is_available():
        print(f"NVIDIA GPU (CUDA) is active: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Apple Silicon GPU (MPS) is active.")
        return torch.device("mps")
    print("GPU not available; defaulting to CPU.")
    return torch.device("cpu")


def normalize(image):
    """Must match the robust per-frame normalization used during training."""
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, (1, 99))
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0, 1).astype(np.float32)


def dice_score(binary_prediction, ground_truth, eps=1e-6):
    prediction = binary_prediction.astype(bool)
    target = ground_truth.astype(bool)
    return (2 * np.logical_and(prediction, target).sum() + eps) / (prediction.sum() + target.sum() + eps)


def iou_score(binary_prediction, ground_truth, eps=1e-6):
    prediction = binary_prediction.astype(bool)
    target = ground_truth.astype(bool)
    return (np.logical_and(prediction, target).sum() + eps) / (np.logical_or(prediction, target).sum() + eps)


def load_tiff_annotations():
    """Load annotations using the same folder/frame convention as training."""
    annotations = {}
    if not ANNOTATION_DIR.exists():
        return annotations
    for folder in ANNOTATION_DIR.iterdir():
        if not folder.is_dir():
            continue
        masks = {}
        for mask_file in folder.glob("*.tif*"):
            match = re.search(r"(\d+)\.tif{1,2}$", mask_file.name, flags=re.IGNORECASE)
            if match is not None:
                masks[int(match.group(1))] = (np.squeeze(tiff.imread(mask_file)) > 0)
        if masks:
            annotations[folder.name] = masks
    return annotations


def save_qualitative_figure(rows, file_path, maximum=8):
    """Create image / annotation / prediction panels from annotated test frames."""
    rows = rows[:maximum]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 4 * len(rows)), squeeze=False)
    for ax_row, row in zip(axes, rows):
        ax_row[0].imshow(row["image"], cmap="gray")
        ax_row[0].set_title(f"{row['recording']} – frame {row['frame']}: input")
        ax_row[1].imshow(row["ground_truth"], cmap="gray", vmin=0, vmax=1)
        ax_row[1].set_title("Manual annotation")
        ax_row[2].imshow(row["prediction"], cmap="gray", vmin=0, vmax=1)
        ax_row[2].set_title(f"UNet prediction | Dice {row['dice']:.3f}")
        for axis in ax_row:
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(file_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


# ============================== Load model ======================================
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {CHECKPOINT_PATH}. "
        "Run train_unet_nd2_compatible.py first, or adjust CHECKPOINT_PATH."
    )

device = device_for_inference()
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
base_channels = checkpoint.get("base_channels", 32)
model = UNet(base=base_channels).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')} | "
      f"validation Dice: {checkpoint.get('val_dice', 'unknown')}")

# =================== Existing ND2 + darkframe data preparation ==================
with open("access.txt", "r", encoding="utf-8") as file:
    content = [line.strip() for line in file]
if len(content) < 2:
    raise ValueError("access.txt needs the ND2 directory in content[1].")

data_directory = Path.cwd() / content[1].lstrip("/\\")
datapath = sorted(data_directory.glob("*.nd2"))
if not datapath:
    raise FileNotFoundError(f"No ND2 files found in {data_directory}.")

annotations = load_tiff_annotations()
PREDICTION_DIR.mkdir(exist_ok=True)
metric_rows = []
figure_rows = []

# ============================= Inference + export ===============================
with torch.no_grad():
    for nd2_file in datapath:
        recording = nd2_file.stem
        print(f"Processing {nd2_file.name} ...")

        # Kept identical in principle to your Cellpose preparation.
        raw = nd2.imread(nd2_file).astype(np.float32)
        new_img = Background().darkfield_assignment(nd2_file)[0]["raw_minus_dark"]
        frames = np.asarray(new_img, dtype=np.float32)
        if frames.ndim == 2:
            frames = frames[None]
        if frames.ndim == 4 and frames.shape[1] == 1:
            frames = frames[:, 0]
        if frames.ndim != 3:
            raise ValueError(f"{nd2_file.name}: expected frames shaped (T, Y, X), got {frames.shape}.")

        recording_output = PREDICTION_DIR / recording
        recording_output.mkdir(exist_ok=True)

        # Batch inference protects GPU/CPU memory for long time series.
        for start in range(0, len(frames), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(frames))
            images = np.stack([normalize(frame) for frame in frames[start:stop]])
            image_tensor = torch.from_numpy(images[:, None]).to(device)
            probabilities = torch.sigmoid(model(image_tensor)).cpu().numpy()[:, 0]

            for local_index, probability in enumerate(probabilities):
                frame_idx = start + local_index
                binary_mask = probability >= THRESHOLD
                instance_mask, _ = label(binary_mask)

                if SAVE_PROBABILITY_MAPS:
                    tiff.imwrite(recording_output / f"frame_{frame_idx:04d}_probability.tif", probability.astype(np.float32))
                if SAVE_BINARY_MASKS:
                    tiff.imwrite(recording_output / f"frame_{frame_idx:04d}_binary.tif", binary_mask.astype(np.uint8))
                if SAVE_INSTANCE_MASKS:
                    tiff.imwrite(recording_output / f"frame_{frame_idx:04d}_instances.tif", instance_mask.astype(np.uint16))

                # Evaluate only a frame that has a manual annotation, never treat
                # an unannotated frame as an empty ground-truth mask.
                ground_truth = annotations.get(recording, {}).get(frame_idx)
                if ground_truth is not None:
                    if ground_truth.shape != binary_mask.shape:
                        raise ValueError(
                            f"{recording}, frame {frame_idx}: annotation {ground_truth.shape} and "
                            f"prediction {binary_mask.shape} have different shapes."
                        )
                    dice = dice_score(binary_mask, ground_truth)
                    iou = iou_score(binary_mask, ground_truth)
                    row = {"recording": recording, "frame": frame_idx, "dice": dice, "iou": iou}
                    metric_rows.append(row)
                    figure_rows.append({**row, "image": images[local_index], "ground_truth": ground_truth,
                                        "prediction": binary_mask})

# =============================== Test report ====================================
if metric_rows:
    csv_path = PREDICTION_DIR / "annotated_test_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["recording", "frame", "dice", "iou"])
        writer.writeheader()
        writer.writerows(metric_rows)
    mean_dice = float(np.mean([row["dice"] for row in metric_rows]))
    mean_iou = float(np.mean([row["iou"] for row in metric_rows]))
    figure_path = PREDICTION_DIR / "qualitative_examples.png"
    save_qualitative_figure(figure_rows, figure_path)
    print(f"Evaluated {len(metric_rows)} manually annotated frames.")
    print(f"Mean Dice: {mean_dice:.4f} | Mean IoU: {mean_iou:.4f}")
    print(f"Metrics: {csv_path}")
    print(f"Example comparison figure: {figure_path}")
else:
    print("No matching TIFF annotations found; exported predictions only.")

print(f"All prediction files saved below: {PREDICTION_DIR}")
