
import numpy as np
import matplotlib.pyplot as plt
from cellpose import metrics, models
from pathlib import Path
import re
import nd2
import numpy as np
import tifffile as tiff
import torch



# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
with open("access.txt", "r", encoding="utf-8") as file:
    content = [line.strip() for line in file]

data = {}
datapath = list(Path(str(Path.cwd()) + content[1]).glob("*.nd2"))
for _, file in enumerate(datapath):
    data[str(file).split("/")[-1].split(".nd2")[0]] = nd2.imread(file).astype(np.float32) # converting to array


maskpath = [folder for folder in Path(str(Path.cwd()) + content[2]).iterdir() if folder.is_dir()]

mask_files = {}

for index, folder in enumerate(maskpath):
    # for _, file in enumerate(maskpath[folder].glob("")):
    mask_files[str(folder).split("/")[-1]] = list(maskpath[index].glob("*"))

output = str(Path.cwd()) + "/cellpose_training"  

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



frames_to_test = [0, 160, 220, 320]
channels = [0, 0]  # Grayscale / single channel

MODEL_PATH = str(Path.cwd()) + "/cellpose_models/models/cellpose_1791574668.9789433"

# Select the available device
if torch.cuda.is_available():
    device = torch.device("cuda")
    use_gpu = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    use_gpu = True
else:
    device = torch.device("cpu")
    use_gpu = False

print("Device:", device)

# Load your trained Cellpose model
model = models.CellposeModel(
    gpu=use_gpu,
    device=device,
    pretrained_model=MODEL_PATH
)

print("Model class:", type(model))
print("Model network:", type(model.net))
print("GPU enabled:", use_gpu)
print("Device:", device)

model = models.CellposeModel(
    gpu=False,
    model_type="livecell"
)


"""
masks : list of 2D arrays or single 3D array
Labelled image, where 0=no masks; 1,2,...=mask labels;

flows : list of lists 2D arrays or list of 3D arrays
flows[k][0] = XY flow in HSV 0-255; flows[k][1] = XY flows at each pixel; flows[k][2] = cell probability (if > cellprob_threshold, pixel used for dynamics); flows[k][3] = final pixel locations after Euler integration;

styles : list of 1D arrays of length 256 or single 1D array
Style vector containing only zeros. Retained for compaibility with CP3.
"""


for keys in data.keys():
    for index in frames_to_test:
        masks, flows, styles = model.eval(data[keys][index], channels = channels)
        labels_masked = np.ma.masked_where(masks == 0, masks)
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.imshow(data[keys][index, 0], cmap="inferno", vmax = data[keys][index, 0].max()*0.8)
        ax.imshow(
            labels_masked,
            cmap="nipy_spectral",
            interpolation="nearest",
            alpha=0.4
        )

        ax.set_title("Cellpose segmentation")
        ax.axis("off")

        plt.tight_layout()
        fig.savefig(str(Path.cwd()) + "/result/cellpose" + f"{keys}_index{index}.png", dpi=300, bbox_inches="tight")

        plt.show()

