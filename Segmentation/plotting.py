import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import nd2
import re 
from pathlib import Path
from cellpose import models, metrics
from training import Trainingpreperation  # Your data preparation class

# Set environment fallback if needed
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

# -------------------------------------------------------------
# 1. CONFIGURATION & PATHS
# -------------------------------------------------------------
MODEL_PATH = "/u/hanguy/Segmentation_UI/cellpose_models/models/cellpose_1790687402.9245734"  # Path to saved model file
ACCESS_FILE = "access.txt"

# -------------------------------------------------------------
# 2. DEVICE SETUP
# -------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    use_gpu = True
    print(f"CUDA active: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    use_gpu = True
    print("Apple Silicon GPU (MPS) active.")
else:
    device = torch.device("cpu")
    use_gpu = False
    print("Defaulting to CPU.")

# -------------------------------------------------------------
# 3. LOAD TEST DATA
# -------------------------------------------------------------
with open(ACCESS_FILE, "r", encoding="utf-8") as file:
    content = [line.strip() for line in file]

data = {}
datapath = list(Path(str(Path.cwd()) + content[1]).glob("*.nd2"))
for file in datapath:
    data[str(file).split("/")[-1].split(".nd2")[0]] = nd2.imread(file).astype(np.float32)

output_folder = Path.cwd() / "cellpose_training"
annotation = {}
frameindex = {}

for folder in output_folder.iterdir():
    if folder.is_dir():
        listig = list(folder.glob("*"))
        listholder = {}
        placeholder = []
        for i in listig:
            frame_num = int(re.findall(r"(\d+)\.tif$", str(i).split("/")[-1])[0])
            listholder[frame_num] = tiff.imread(i)
            placeholder.append(frame_num)

        key = str(folder).split("/")[-1]
        annotation[key] = listholder
        frameindex[key] = sorted(placeholder)

# Retrieve only the test frames split
prep = Trainingpreperation()
_, _, test_frames = prep.split_frames_by_key(frameindex)

test_images, test_masks = [], []
for key, frames in test_frames.items():
    for frame_idx in frames:
        test_images.append(data[key][:, 0][frame_idx])
        test_masks.append(annotation[key][frame_idx])

print(f"Loaded {len(test_images)} test samples.")

# -------------------------------------------------------------
# 4. LOAD TRAINED MODEL
# -------------------------------------------------------------
# Pass model_path to preload your custom weights into CellposeModel
model = models.CellposeModel(
    gpu=use_gpu, 
    device=device, 
    pretrained_model=MODEL_PATH
)

# -------------------------------------------------------------
# 5. INFERENCE & EVALUATION
# -------------------------------------------------------------
channels = [0, 0]  # [0, 0] for single-channel/grayscale
masks_pred, flows, styles = model.eval(test_images, channels=channels)

# Define IoU thresholds
iou_thresholds = np.arange(0.5, 1.0, 0.05)

# Calculate precision metrics
ap, tp, fp, fn = metrics.average_precision(
    test_masks, masks_pred, threshold=iou_thresholds
)

mean_ap = ap.mean(axis=0)
total_tp = tp.sum(axis=0)
total_fp = fp.sum(axis=0)
total_fn = fn.sum(axis=0)

precision = total_tp / (total_tp + total_fp + 1e-8)
recall = total_tp / (total_tp + total_fn + 1e-8)

# -------------------------------------------------------------
# 6. PLOTTING RESULTS
# -------------------------------------------------------------
def plot_prediction_samples(images, true_masks, pred_masks, num_samples=3):
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    for i in range(min(num_samples, len(images))):
        axes[i, 0].imshow(images[i], cmap="gray")
        axes[i, 0].set_title(f"Sample {i+1}: Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(true_masks[i], cmap="nipy_spectral")
        axes[i, 1].set_title(f"Sample {i+1}: Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_masks[i], cmap="nipy_spectral")
        axes[i, 2].set_title(f"Sample {i+1}: Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()

plot_prediction_samples(test_images, test_masks, masks_pred, num_samples=3)

# Dashboard plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(iou_thresholds, mean_ap, marker="o", color="#2b5c8f", linewidth=2, label="mAP")
axes[0].set_xlabel("IoU Threshold")
axes[0].set_ylabel("Mean Average Precision (mAP)")
axes[0].set_title("Model Precision across IoU Thresholds")
axes[0].set_ylim([0, 1.05])
axes[0].grid(True, linestyle="--", alpha=0.6)
axes[0].legend()

threshold_indices = [0, 5]  # IoU 0.50 and 0.75
labels = [f"IoU {iou_thresholds[i]:.2f}" for i in threshold_indices]
x = np.arange(len(labels))
width = 0.25

axes[1].bar(x - width, [total_tp[i] for i in threshold_indices], width, label="True Positives", color="#2ca02c")
axes[1].bar(x, [total_fp[i] for i in threshold_indices], width, label="False Positives", color="#d62728")
axes[1].bar(x + width, [total_fn[i] for i in threshold_indices], width, label="False Negatives", color="#ff7f0e")
axes[1].set_ylabel("Count")
axes[1].set_title("Detection Breakdown (TP, FP, FN)")
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"{'IoU Threshold':<15} | {'mAP':<8} | {'Precision':<10} | {'Recall':<8}")
print("-" * 50)
for i, thresh in enumerate(iou_thresholds):
    print(f"{thresh:<15.2f} | {mean_ap[i]:<8.3f} | {precision[i]:<10.3f} | {recall[i]:<8.3f}")