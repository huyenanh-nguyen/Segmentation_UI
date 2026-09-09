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
