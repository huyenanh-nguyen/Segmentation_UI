import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import torch
import re
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import zipfile
from zipfile import BadZipFile
import nd2
from torch.utils.data import DataLoader
import numpy as np
import tifffile as tiff
from roifile import ImagejRoi
from skimage.draw import polygon
from scipy.ndimage import (
    distance_transform_edt,
    center_of_mass
)
from scipy import ndimage as ndi
from cellpose import models, io
from training import Trainingpreperation
import random

# [Generating Annotations Masks]_________________

with open("access.txt", "r", encoding="utf-8") as file:
    content = [line.strip() for line in file]

data = {}
datapath = list(Path(content[1]).glob("*.nd2"))

for _, file in enumerate(datapath):
    data[str(file).split("/")[-1].split(".nd2")[0]] = nd2.imread(file).astype(np.float32) # converting to array


maskpath = [folder for folder in Path(content[2]).iterdir() if folder.is_dir()]

mask_files = {}

for index, folder in enumerate(maskpath):
    # for _, file in enumerate(maskpath[folder].glob("")):
    mask_files[str(folder).split("/")[-1]] = list(maskpath[index].glob("*"))

output = str(Path.cwd()) + "/cellpose_training"  

# #[converting .zip and .roi files to .tiff]
# for _, name in enumerate(mask_files):
#     converting = [Trainingpreperation().process_rois_to_tiff(str(p), output, 600, 600, name) for p in mask_files[name]]


# [Annotation path]_________________

"""
annotation dictionary: Keys are the name of the folders
"""

annotation = {} # where the mask is
frameindex = {}

for index, folder in enumerate(Path(output).iterdir()):
    if folder.is_dir():
        listig = list(folder.glob("*"))
        listholder = {}
        placeholder = []
        for i in listig:
            listholder[int(re.findall(r"(\d+)\.tif$", str(i).split("/")[-1])[0])] = tiff.imread(i)
            placeholder.append(int(re.findall(r"(\d+)\.tif$", str(i).split("/")[-1])[0]))

        annotation[str(folder).split("/")[-1]] = listholder # annotation as arrays
        frameindex[str(folder).split("/")[-1]] = sorted(placeholder)



# for _, folder in enumerate(annotation.keys()):
#     placeholder = []
#     for index, name in enumerate(annotation[folder]):
#         placeholder.append(int(re.findall(r"(\d+)\.tif$", str(name).split("/")[-1])[0]))
#     frameindex[folder] = sorted(placeholder)


# [training the Cellpose model]_________________

# now distributing the frames to training, test and validation data.

train_frames, val_frames, test_frames = Trainingpreperation().split_frames_by_key(frameindex)

print(test_frames)


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Apple Silicon GPU (MPS) is active.")
else:
    device = torch.device("cpu")
    print("MPS not available; defaulting to CPU.")






# Initialize lists to hold images and masks
train_images, train_masks = [], []
val_images, val_masks = [], []
test_images, test_masks = [], []

prep = Trainingpreperation()

# [Data Augmentation to extend more Datatraningsdata]
# Set how many augmented variations you want to generate per frame
NUM_AUGMENTATIONS_PER_FRAME = 2  # Generates original + N augmented versions

for key, frames in train_frames.items():

    for frame_idx in frames:
        # Append original unaugmented pair
        train_images.append(data[key][:,0][frame_idx])
        train_masks.append(annotation[key][frame_idx])

        # Generate N augmented copies
        for _ in range(NUM_AUGMENTATIONS_PER_FRAME):
            aug_img, aug_mask = prep.spatial_augmentation(data[key][:,0][frame_idx], annotation[key][frame_idx])
            aug_img = prep.pixelaugmentation(aug_img)

            train_images.append(aug_img)
            train_masks.append(aug_mask)

# -------------------------------------------------------------
# 2. PROCESS VALIDATION SET (UNTOUCHED / NO AUGMENTATION)
# -------------------------------------------------------------
for key, frames in val_frames.items():
    for frame_idx in frames:
        val_images.append(data[key][:,0][frame_idx])
        val_masks.append(annotation[key][frame_idx])

# -------------------------------------------------------------
# 3. PROCESS TEST SET (UNTOUCHED / NO AUGMENTATION)
# -------------------------------------------------------------
for key, frames in test_frames.items():
    for frame_idx in frames:
        test_images.append(data[key][:,0][frame_idx])
        test_masks.append(annotation[key][frame_idx])

print(f"Total training samples: {len(train_images)}")
print(f"Total validation samples: {len(val_images)}")
print(f"Total test samples: {len(test_images)}")



# import torch
# from cellpose import models, train

# # Device check
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print("Apple Silicon GPU (MPS) is active.")
# else:
#     device = torch.device("cpu")
#     print("MPS not available; defaulting to CPU.")

# # 1. Initialize Cellpose model
# # Choose base model type, e.g., 'cyto3', 'cyto2', or 'nuclei'
# model = models.CellposeModel(gpu=True, device=device, model_type="cyto3")

# # 2. Set channel configurations
# # For single-channel / grayscale images, use [0, 0]
# # For multi-channel (e.g. Red/Green fluorescence): [1, 2] (1=cytoplasm, 2=nucleus)
# channels = [0, 0]

# # 3. Start Model Training
# model_path = train.train_seg(
#     model.net,
#     train_data=train_images,
#     train_labels=train_masks,
#     test_data=val_images,
#     test_labels=val_masks,
#     channels=channels,
#     batch_size=8,  # M2 Pro 16GB/32GB RAM handles batch size 8-16 easily
#     n_epochs=100,  # Recommended 100-500 depending on dataset size
#     learning_rate=0.1,
#     weight_decay=0.0001,
#     save_path="./cellpose_models",
# )

# print(f"Training completed successfully! Saved model path: {model_path}")


# from cellpose import metrics

# # Run prediction on test images
# masks_pred, flows, styles = model.eval(test_images, channels=channels)

# # Evaluate average precision at IoU thresholds 0.5 to 0.95
# ap, tp, fp, fn = metrics.average_precision(
#     test_masks, masks_pred, threshold=[0.5, 0.75]
# )

# print(
#     f"Mean Average Precision (mAP @ IoU 0.5): {ap[:, 0].mean():.3f}"
# )  # standard IoU=0.5
# print(f"Mean Average Precision (mAP @ IoU 0.75): {ap[:, 1].mean():.3f}")