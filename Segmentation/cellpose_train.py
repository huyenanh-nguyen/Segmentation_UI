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
from cellpose import models, io
from training import Trainingpreperation


with open("access.txt", "r", encoding="utf-8") as file:
    content = [line.strip() for line in file]

datapath = Path(content[1])
maskpath = [folder for folder in Path(content[2]).iterdir() if folder.is_dir()]

mask_files = {}

for folder in range(len(maskpath)):
    # for _, file in enumerate(maskpath[folder].glob("")):
    mask_files[folder] = list(maskpath[folder].glob("*"))

output = str(Path.cwd()) + "/cellpose_training"  
print(output)

for i in range(len(mask_files)):
    print(mask_files[i][0])
    converting = [Trainingpreperation().process_rois_to_tiff(str(p), output, 600, 600, str(i)) for p in mask_files[i]]

