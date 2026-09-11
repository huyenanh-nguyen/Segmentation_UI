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

# ================================================================================================================================================================================================================================================
# Torch on GPU instead of CPU
# ================================================================================================================================================================================================================================================

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0" # IMPORTANT: Disable MPS CPU fallback

# For Apple M2 pro Chip
if not torch.backends.mps.is_available():
    raise RuntimeError(
        "MPS is not available! "
        "Training stopped so the CPU will not be used."
    )

device = torch.device("mps")

print(f"Using device: {device}")


# # ================================================================================================================================================================================================================================================
# # Data
# # ================================================================================================================================================================================================================================================

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


# ================================================================================================================================================================================================================================================
# FUNCTION: Add one ROI to a mask
# ================================================================================================================================================================================================================================================

def add_roi_to_mask(roi, mask, label=1):

    coords = roi.coordinates()

    if coords is None:
        return False

    x = coords[:, 0]
    y = coords[:, 1]

    rr, cc = polygon(y, x, shape=mask.shape)

    mask[rr, cc] = label

    return True


# ================================================================================================================================================================================================================================================
# Converting ROI files into mask.tiff, that I draw in ImageJ for the Training
# ================================================================================================================================================================================================================================================

def converting_zip_to_tiff(folderpath, outputfolder):

    for zip_path in folderpath.glob("*.zip"):

        print(f"\nProcessing ZIP: {zip_path.name}")

        try:
            with zipfile.ZipFile(zip_path) as z:
                roi_names = [
                    name for name in z.namelist()
                    if name.lower().endswith(".roi")
                ]

        except BadZipFile:
            print(f"  Skipping {zip_path.name}: not a valid ZIP file")
            continue

        print(f"  Found {len(roi_names)} ROIs")

        if len(roi_names) == 0:
            print("  No ROI files, skipping.")
            continue

        mask_array = np.zeros((height, width), dtype=np.uint16)

        with zipfile.ZipFile(zip_path) as z:

            label = 1

            for roi_name in roi_names:

                try:
                    with z.open(roi_name) as f:
                        roi = ImagejRoi.frombytes(f.read())

                    success = add_roi_to_mask(
                        roi,
                        mask_array,
                        label
                    )

                    if success:
                        label += 1
                    else:
                        print(f"  Skipping {roi_name}")

                except Exception as e:
                    print(f"  Error reading {roi_name}: {e}")

        # zip_path.stem automatically removes ".zip"
        output_mask = outputfolder / f"{zip_path.stem.replace('.roi', '')}.tif"

        tiff.imwrite(output_mask, mask_array)

        print(f"  Saved mask: {output_mask.name}")
        print(f"  Labels: {np.unique(mask_array)}")

        return None


def converting_roi_to_tiff(folderpath, outputfolder):
    for roi_path in folderpath.glob("*.roi"):

        print(f"\nProcessing ROI: {roi_path.name}")

        mask_array = np.zeros((height, width), dtype=np.uint16)

        try:

            # Read the individual ROI
            roi = ImagejRoi.fromfile(roi_path)

            success = add_roi_to_mask(
                roi,
                mask_array,
                label=1
            )

            if not success:
                print("  Could not extract coordinates, skipping.")
                continue

            # roi_path.stem automatically removes ".roi"
            output_mask = outputfolder / f"{roi_path.stem}.tif"

            tiff.imwrite(output_mask, mask_array)

            print(f"  Saved mask: {output_mask.name}")
            print(f"  Labels: {np.unique(mask_array)}")

        except Exception as e:
            print(f"  Error processing {roi_path.name}: {e}")

    return None


# ================================================================================================================================================================================================================================================
# Creating UNET
# ================================================================================================================================================================================================================================================

"""
What is an UNET? (Source: Wikipedia, https://en.wikipedia.org/wiki/U-Net)

The U-Net architecture stems from the so-called "fully convolutional network". 
The network consists of a contracting path and an expansive path, which gives it the u-shaped architecture. 
The contracting path is a typical convolutional network that consists of repeated application of convolutions, each followed by a rectified linear unit (ReLU) and a max pooling operation. 
During the contraction, the spatial information is reduced while feature information is increased. 
The expansive pathway combines the feature and spatial information through a sequence of up-convolutions and concatenations with high-resolution features from the contracting path
"""

"""
What is convolution? (Source: Wikipedia, https://en.wikipedia.org/wiki/Convolution)
In mathematics (in particular, functional analysis), convolution is a mathematical operation on two functions f and g that produces a third function f∗g, 
as the integral of the product of the two functions after one is reflected about the y-axis and shifted.
The term convolution refers to both the resulting function and to the process of computing it.
"""


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


"""
Project arbeit:
Investigation of methode for biological Image processing (was es für methoden gibt, wie algorithmische oder NN oder pretrained models)
oder
Application of cellular method

Master thesis:
Development pipline of Interleukin cells with method NN
- test machen
- annotated data genug davon
- data augmentation
- braucht man mehr data
- verschiedene movies davon 10 frames, je mehr verschiedene daten desto besser
- eine zelle rausnehmen und die verändern, modifzieren so rotieren, verschwimmen und die verfälschen -> 10 verschiedene transformationen
- yolo11
- wenn man eine datenart trainiert dann wird es overfitten
trainingsdaten (20%) und testdaten (80%)
"""
