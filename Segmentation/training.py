import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import torch
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


def normalizing_image(img, low, high):
    """normalizing images, to merge same values

    Args:
        img (array): image
        low (float): lowest percentile
        high (float): highes percentile

    Returns:
        Array: the image is normalized
    """
    data_min = np.percentile(img, low, axis = (1,2), keepdims = True)
    data_max = np.percentile(img, high, axis = (1,2), keepdims = True)
    scaled_data = np.clip((img - data_min) / ( data_max - data_min + 1e-8), 0, 1)
    return scaled_data

nd2_path = "images/20221216 3nM 231-A8_NEMO_TRAF6_1um_38mol 001.nd2"
nd2_array = nd2.imread(nd2_path)
dark_400 = tiff.imread("images/20210531 Darkfield 400ms.tif").astype(np.float32)
dark_60 = tiff.imread("images/20210531 Darkfield 60ms.tif").astype(np.float32)

rfp = np.clip(nd2_array[:,0].astype(np.float32) - dark_400, 0, None) # substracting dark_frame
gfp = np.clip(nd2_array[:,1].astype(np.float32) - dark_60, 0, None)
bf = np.clip(nd2_array[:,2].astype(np.float32) - dark_60, 0, None)
merged_ind_norm = normalizing_image(rfp, 1, 99.8) + normalizing_image(gfp, 1, 99.8)
mask = "/Users/huyenanh/Documents/Anhs Uni/Master Thesis/masks/20221216 3nM 231-A8_NEMO_TRAF6_1um_38mol 001"


input_folder = Path(mask)
output_folder = Path(
    "/Users/huyenanh/git_repos/Segmentation_UI/cellpose_training/rollballtrain"
)

output_folder.mkdir(exist_ok=True)

mask_tif = str(output_folder)

# height = 600
# width = 600


# # ============================================================
# # FUNCTION: Add one ROI to a mask
# # ============================================================

# def add_roi_to_mask(roi, mask, label=1):

#     coords = roi.coordinates()

#     if coords is None:
#         return False

#     x = coords[:, 0]
#     y = coords[:, 1]

#     rr, cc = polygon(y, x, shape=mask.shape)

#     mask[rr, cc] = label

#     return True


# # ============================================================
# # PROCESS ZIP FILES
# # ============================================================

# for zip_path in input_folder.glob("*.zip"):

#     print(f"\nProcessing ZIP: {zip_path.name}")

#     try:
#         with zipfile.ZipFile(zip_path) as z:
#             roi_names = [
#                 name for name in z.namelist()
#                 if name.lower().endswith(".roi")
#             ]

#     except BadZipFile:
#         print(f"  Skipping {zip_path.name}: not a valid ZIP file")
#         continue

#     print(f"  Found {len(roi_names)} ROIs")

#     if len(roi_names) == 0:
#         print("  No ROI files, skipping.")
#         continue

#     mask_array = np.zeros((height, width), dtype=np.uint16)

#     with zipfile.ZipFile(zip_path) as z:

#         label = 1

#         for roi_name in roi_names:

#             try:
#                 with z.open(roi_name) as f:
#                     roi = ImagejRoi.frombytes(f.read())

#                 success = add_roi_to_mask(
#                     roi,
#                     mask_array,
#                     label
#                 )

#                 if success:
#                     label += 1
#                 else:
#                     print(f"  Skipping {roi_name}")

#             except Exception as e:
#                 print(f"  Error reading {roi_name}: {e}")

#     # zip_path.stem automatically removes ".zip"
#     output_mask = output_folder / f"{zip_path.stem.replace('.roi', '')}.tif"

#     tiff.imwrite(output_mask, mask_array)

#     print(f"  Saved mask: {output_mask.name}")
#     print(f"  Labels: {np.unique(mask_array)}")


# # ============================================================
# # PROCESS INDIVIDUAL ROI FILES
# # ============================================================

# for roi_path in input_folder.glob("*.roi"):

#     print(f"\nProcessing ROI: {roi_path.name}")

#     mask_array = np.zeros((height, width), dtype=np.uint16)

#     try:

#         # Read the individual ROI
#         roi = ImagejRoi.fromfile(roi_path)

#         success = add_roi_to_mask(
#             roi,
#             mask_array,
#             label=1
#         )

#         if not success:
#             print("  Could not extract coordinates, skipping.")
#             continue

#         # roi_path.stem automatically removes ".roi"
#         output_mask = output_folder / f"{roi_path.stem}.tif"

#         tiff.imwrite(output_mask, mask_array)

#         print(f"  Saved mask: {output_mask.name}")
#         print(f"  Labels: {np.unique(mask_array)}")

#     except Exception as e:
#         print(f"  Error processing {roi_path.name}: {e}")


import os

# ============================================================
# IMPORTANT: Disable MPS CPU fallback
# ============================================================

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import numpy as np
import tifffile as tiff

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from scipy.ndimage import (
    distance_transform_edt,
    center_of_mass
)

from skimage.feature import peak_local_max


# ============================================================
# DEVICE — FORCE APPLE SILICON GPU
# ============================================================

if not torch.backends.mps.is_available():
    raise RuntimeError(
        "MPS is not available! "
        "Training stopped so the CPU will not be used."
    )

device = torch.device("mps")

print(f"Using device: {device}")


# ============================================================
# UNET
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


class SmallUNet(nn.Module):

    def __init__(
        self,
        in_channels=5,
        out_channels=2
    ):

        super().__init__()

        # --------------------
        # Encoder
        # --------------------

        self.enc1 = DoubleConv(
            in_channels,
            32
        )

        self.enc2 = DoubleConv(
            32,
            64
        )

        self.enc3 = DoubleConv(
            64,
            128
        )

        self.pool = nn.MaxPool2d(2)


        # --------------------
        # Bottleneck
        # --------------------

        self.bottleneck = DoubleConv(
            128,
            256
        )


        # --------------------
        # Decoder
        # --------------------

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            256,
            128
        )


        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            128,
            64
        )


        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            64,
            32
        )


        # --------------------
        # Output
        # --------------------

        self.output = nn.Conv2d(
            32,
            out_channels,
            kernel_size=1
        )


    def forward(self, x):

        # Encoder

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )


        # Bottleneck

        b = self.bottleneck(
            self.pool(e3)
        )


        # Decoder

        d3 = self.up3(b)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)


        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)


        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)


        return self.output(d1)


# ============================================================
# CREATE TEMPORAL WINDOW
# ============================================================

def get_temporal_window(
    images,
    t,
    radius=2
):

    frames = []

    num_frames = len(images)

    for offset in range(-radius, radius + 1):

        frame_index = t + offset

        # Handle beginning/end of movie
        frame_index = np.clip(
            frame_index,
            0,
            num_frames - 1
        )

        frames.append(
            images[frame_index]
        )

    # Shape:
    # (5, height, width)

    return np.stack(
        frames,
        axis=0
    )


# ============================================================
# CREATE CENTER MAP
# ============================================================

def create_center_map(
    instance_mask,
    sigma=5
):

    """
    Creates a center target map.

    Each cell gets a Gaussian-like center region.
    """

    from scipy.ndimage import gaussian_filter

    center_map = np.zeros(
        instance_mask.shape,
        dtype=np.float32
    )

    labels = np.unique(instance_mask)

    labels = labels[
        labels != 0
    ]

    for label in labels:

        cell = (
            instance_mask == label
        )

        cy, cx = center_of_mass(cell)

        cy = int(round(cy))
        cx = int(round(cx))

        if (
            cy < 0
            or cy >= center_map.shape[0]
            or cx < 0
            or cx >= center_map.shape[1]
        ):
            continue

        center_map[cy, cx] = 1.0


    # Blur center points

    center_map = gaussian_filter(
        center_map,
        sigma=sigma
    )


    # Normalize

    max_value = center_map.max()

    if max_value > 0:

        center_map = (
            center_map / max_value
        )


    return center_map.astype(
        np.float32
    )


# ============================================================
# DATASET
# ============================================================

class CellDataset(Dataset):

    def __init__(
        self,
        images,
        annotation_frames,
        mask_folder,
        patch_size=256,
        samples_per_frame=100,
        temporal_radius=2
    ):

        self.images = images

        self.annotation_frames = annotation_frames

        self.mask_folder = Path(mask_folder)

        self.patch_size = patch_size

        self.samples_per_frame = samples_per_frame

        self.temporal_radius = temporal_radius


        # ----------------------------------------------------
        # LOAD MASKS
        # ----------------------------------------------------

        self.masks = {}

        print("\nLoading masks:")

        for frame in annotation_frames:

            # IMPORTANT:
            #
            # frame index 0 → mask001.tif
            #
            # frame index 10 → mask011.tif

            mask_number = frame + 1

            mask_path = (
                self.mask_folder
                / f"mask{mask_number:03d}.tif"
            )

            if not mask_path.exists():

                raise FileNotFoundError(
                    f"\nMask not found!\n"
                    f"Frame index: {frame}\n"
                    f"Expected mask: {mask_path}"
                )

            print(
                f"Frame {frame} → "
                f"{mask_path.name}"
            )

            mask = tiff.imread(
                mask_path
            )

            self.masks[frame] = mask.astype(
                np.int32
            )


    def __len__(self):

        return (
            len(self.annotation_frames)
            * self.samples_per_frame
        )


    def __getitem__(self, index):

        # ----------------------------------------------------
        # Select frame
        # ----------------------------------------------------

        frame = self.annotation_frames[
            index % len(self.annotation_frames)
        ]


        # ----------------------------------------------------
        # Get temporal input
        # ----------------------------------------------------

        temporal_input = get_temporal_window(
            self.images,
            frame,
            radius=self.temporal_radius
        )


        # Shape:
        #
        # (5, H, W)


        instance_mask = self.masks[frame]


        # ----------------------------------------------------
        # Foreground target
        # ----------------------------------------------------

        foreground = (
            instance_mask > 0
        ).astype(np.float32)


        # ----------------------------------------------------
        # Center target
        # ----------------------------------------------------

        centers = create_center_map(
            instance_mask
        )


        # ----------------------------------------------------
        # RANDOM PATCH
        # ----------------------------------------------------

        height = temporal_input.shape[1]

        width = temporal_input.shape[2]


        if (
            height < self.patch_size
            or width < self.patch_size
        ):

            raise ValueError(
                f"Patch size {self.patch_size} "
                f"is larger than image size "
                f"{height} x {width}"
            )


        y_start = np.random.randint(
            0,
            height - self.patch_size + 1
        )

        x_start = np.random.randint(
            0,
            width - self.patch_size + 1
        )


        # Input patch

        x = temporal_input[
            :,
            y_start:y_start + self.patch_size,
            x_start:x_start + self.patch_size
        ]


        # Foreground patch

        foreground_patch = foreground[
            y_start:y_start + self.patch_size,
            x_start:x_start + self.patch_size
        ]


        # Center patch

        centers_patch = centers[
            y_start:y_start + self.patch_size,
            x_start:x_start + self.patch_size
        ]


        # ----------------------------------------------------
        # TARGET
        # ----------------------------------------------------

        y = np.stack(
            [
                foreground_patch,
                centers_patch
            ],
            axis=0
        )


        # ----------------------------------------------------
        # CONVERT TO PYTORCH
        # ----------------------------------------------------

        x = torch.from_numpy(
            x.copy()
        ).float()

        y = torch.from_numpy(
            y.copy()
        ).float()


        return x, y


# ============================================================
# LOSS FUNCTIONS
# ============================================================

def dice_loss(
    pred,
    target,
    eps=1e-6
):

    pred = torch.sigmoid(pred)

    intersection = (
        pred * target
    ).sum(
        dim=(1, 2)
    )

    union = (
        pred.sum(dim=(1, 2))
        + target.sum(dim=(1, 2))
    )

    dice = (
        2 * intersection + eps
    ) / (
        union + eps
    )

    return 1 - dice.mean()


def loss_function(
    pred,
    target
):

    # ========================================================
    # FOREGROUND
    # ========================================================

    foreground_bce = (
        F.binary_cross_entropy_with_logits(
            pred[:, 0],
            target[:, 0]
        )
    )

    foreground_dice = dice_loss(
        pred[:, 0],
        target[:, 0]
    )

    foreground_loss = (
        foreground_bce
        + foreground_dice
    )


    # ========================================================
    # CENTERS
    # ========================================================

    center_pred = torch.sigmoid(
        pred[:, 1]
    )

    center_loss = F.mse_loss(
        center_pred,
        target[:, 1]
    )


    # ========================================================
    # TOTAL
    # ========================================================

    return (
        foreground_loss
        + center_loss
    )


# # ============================================================
# # SETTINGS
# # ============================================================

# annotation_frames = [
#     0, 3, 22, 49, 99, 159, 199, 259, 299, 359, 399, 449
# ]

# mask_folder = mask_tif


# patch_size = 256

# samples_per_frame = 100

# batch_size = 2

# epochs = 30

# learning_rate = 1e-4


# # ============================================================
# # CREATE DATASET
# # ============================================================

# dataset = CellDataset(

#     images=merged_ind_norm,

#     annotation_frames=annotation_frames,

#     mask_folder=mask_folder,

#     patch_size=patch_size,

#     samples_per_frame=samples_per_frame,

#     temporal_radius=2
# )


# # ============================================================
# # CHECK ONE SAMPLE
# # ============================================================

# x_test, y_test = dataset[0]

# print("\nDataset check:")

# print(
#     "Input shape:",
#     x_test.shape
# )

# print(
#     "Target shape:",
#     y_test.shape
# )


# # Expected:
# #
# # Input:
# # torch.Size([5, 256, 256])
# #
# # Target:
# # torch.Size([2, 256, 256])


# # ============================================================
# # DATALOADER
# # ============================================================

# loader = DataLoader(

#     dataset,

#     batch_size=batch_size,

#     shuffle=True,

#     num_workers=0
# )


# # ============================================================
# # CREATE MODEL
# # ============================================================

# model = SmallUNet(
#     in_channels=5,
#     out_channels=2
# ).to(device)


# optimizer = torch.optim.AdamW(

#     model.parameters(),

#     lr=learning_rate
# )


# # ============================================================
# # TRAINING
# # ============================================================

# best_loss = float("inf")


# for epoch in range(epochs):

#     model.train()

#     epoch_loss = 0.0


#     print(
#         f"\n{'=' * 60}"
#     )

#     print(
#         f"Starting Epoch "
#         f"{epoch + 1}/{epochs}"
#     )

#     print(
#         f"{'=' * 60}"
#     )


#     for batch_idx, (x, y) in enumerate(loader):

#         # ----------------------------------------------------
#         # Move to Apple GPU
#         # ----------------------------------------------------

#         x = x.to(
#             device,
#             dtype=torch.float32
#         )

#         y = y.to(
#             device,
#             dtype=torch.float32
#         )


#         # ----------------------------------------------------
#         # Forward
#         # ----------------------------------------------------

#         optimizer.zero_grad()

#         pred = model(x)


#         # ----------------------------------------------------
#         # Loss
#         # ----------------------------------------------------

#         loss = loss_function(
#             pred,
#             y
#         )


#         # ----------------------------------------------------
#         # Backpropagation
#         # ----------------------------------------------------

#         loss.backward()

#         optimizer.step()


#         epoch_loss += (
#             loss.item()
#         )


#         print(

#             f"Epoch {epoch + 1}/{epochs} | "

#             f"Batch {batch_idx + 1}/{len(loader)} | "

#             f"Loss: {loss.item():.4f}",

#             end="\r",

#             flush=True
#         )


#     # ========================================================
#     # EPOCH RESULTS
#     # ========================================================

#     average_loss = (
#         epoch_loss / len(loader)
#     )


#     print(
#         f"\n\nEpoch {epoch + 1} finished"
#     )

#     print(
#         f"Average loss: "
#         f"{average_loss:.6f}"
#     )


#     # ========================================================
#     # SAVE BEST MODEL
#     # ========================================================

#     if average_loss < best_loss:

#         best_loss = average_loss


#         checkpoint = {

#             "epoch": epoch,

#             "model_state_dict":
#                 model.state_dict(),

#             "optimizer_state_dict":
#                 optimizer.state_dict(),

#             "loss":
#                 best_loss,

#             "annotation_frames":
#                 annotation_frames,

#             "patch_size":
#                 patch_size,

#             "temporal_radius":
#                 2
#         }


#         torch.save(
#             checkpoint,
#             "best_unet_mps.pth"
#         )


#         print(
#             "⭐ New best model saved!"
#         )


# ============================================================
# FINISHED
# ============================================================

# t = 146
# print("\nTraining finished!")

# print(
#     f"Best loss: {best_loss:.6f}"
# )

# print(
#     "Model saved as:"
# )

# print(
#     "best_unet_mps.pth"
# )


print("Foreground:")
print("min:", foreground.min())
print("max:", foreground.max())
print("mean:", foreground.mean())

print("\nCenters:")
print("min:", centers.min())
print("max:", centers.max())
print("mean:", centers.mean())



# ============================================================
# SETTINGS
# ============================================================

t = 10

foreground_threshold = 0.5
center_threshold = 0.3
min_distance = 20


# ============================================================
# GET TEMPORAL WINDOW
# ============================================================

input_array = get_temporal_window(
    merged_ind_norm,
    t,
    radius=2
)

print("Input shape:", input_array.shape)
print("Input min/max:", input_array.min(), input_array.max())


# ============================================================
# CONVERT TO TORCH
# ============================================================

input_tensor = torch.from_numpy(
    input_array.copy()
).float()

# (5, H, W) -> (1, 5, H, W)

input_tensor = input_tensor.unsqueeze(0)

input_tensor = input_tensor.to(device)


# ============================================================
# MODEL PREDICTION
# ============================================================

import torch
from pathlib import Path

device = torch.device("mps")

# Create the same model architecture
model = SmallUNet(
    in_channels=5,
    out_channels=2
).to(device)

# Path to saved model
model_path = Path("best_unet_mps.pth")

# Load checkpoint
checkpoint = torch.load(model_path, map_location=device)

# Load trained weights
model.load_state_dict(checkpoint["model_state_dict"])

# Evaluation mode
model.eval()

print("Model loaded!")

with torch.no_grad():

    prediction = model(input_tensor)

    prediction = torch.sigmoid(prediction)


# ============================================================
# GET OUTPUTS
# ============================================================

foreground = (
    prediction[0, 0]
    .detach()
    .cpu()
    .numpy()
)

centers = (
    prediction[0, 1]
    .detach()
    .cpu()
    .numpy()
)


print("\nPrediction shape:", prediction.shape)

print(
    "Foreground range:",
    foreground.min(),
    foreground.max()
)

print(
    "Center range:",
    centers.min(),
    centers.max()
)


# ============================================================
# FOREGROUND MASK
# ============================================================

cell_mask = (
    foreground > foreground_threshold
)

print(
    "Foreground pixels:",
    cell_mask.sum()
)


# ============================================================
# FIND CENTERS
# ============================================================

coordinates = peak_local_max(
    centers,
    min_distance=min_distance,
    threshold_abs=center_threshold
)

print(
    "Number of detected centers:",
    len(coordinates)
)


# ============================================================
# CREATE MARKERS
# ============================================================

markers = np.zeros(
    cell_mask.shape,
    dtype=np.int32
)

for label, (cy, cx) in enumerate(
    coordinates,
    start=1
):

    markers[cy, cx] = label


# ============================================================
# WATERSHED
# ============================================================

distance = distance_transform_edt(
    cell_mask
)

labels = watershed(
    -distance,
    markers,
    mask=cell_mask
)

print(
    "Number of segmented cells:",
    labels.max()
)


# ============================================================
# PLOT 1 — ORIGINAL IMAGE
# ============================================================

plt.figure(figsize=(8, 8))

plt.imshow(
    merged_ind_norm[t],
    cmap="gray"
)

plt.title(
    f"Original — frame {t}"
)

plt.axis("off")

plt.show()


# ============================================================
# PLOT 2 — FOREGROUND PROBABILITY
# ============================================================

plt.figure(figsize=(8, 8))

plt.imshow(
    merged_ind_norm[t],
    cmap="gray"
)

plt.imshow(
    foreground,
    cmap="hot",
    alpha=0.6,
    vmin=0,
    vmax=1
)

plt.colorbar(
    label="Foreground probability"
)

plt.title(
    f"UNet foreground — frame {t}"
)

plt.axis("off")

plt.show()


# ============================================================
# PLOT 3 — CENTER PROBABILITY
# ============================================================

plt.figure(figsize=(8, 8))

plt.imshow(
    merged_ind_norm[t],
    cmap="gray"
)

plt.imshow(
    centers,
    cmap="hot",
    alpha=0.6,
    vmin=0,
    vmax=1
)

if len(coordinates) > 0:

    plt.scatter(
        coordinates[:, 1],
        coordinates[:, 0],
        s=30,
        facecolors="none",
        edgecolors="white",
        linewidths=1
    )

plt.colorbar(
    label="Center probability"
)

plt.title(
    f"UNet centers — frame {t}"
)

plt.axis("off")

plt.show()


# ============================================================
# PLOT 4 — FINAL SEGMENTATION OVERLAY
# ============================================================

plt.figure(figsize=(8, 8))

plt.imshow(
    merged_ind_norm[t],
    cmap="gray"
)

labels_masked = np.ma.masked_where(
    labels == 0,
    labels
)

plt.imshow(
    labels_masked,
    cmap="gist_ncar",
    alpha=0.5,
    interpolation="none"
)

if len(coordinates) > 0:

    plt.scatter(
        coordinates[:, 1],
        coordinates[:, 0],
        s=30,
        facecolors="none",
        edgecolors="white",
        linewidths=1
    )

plt.title(
    f"UNet segmentation — frame {t} — "
    f"{labels.max()} cells"
)

plt.axis("off")

plt.show()