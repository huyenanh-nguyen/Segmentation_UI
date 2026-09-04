import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt
import numpy as np
import matplotlib.pyplot as plt

nd2_path = "images/20221216 3nM 231-A8_NEMO_TRAF6_1um_38mol 001.nd2"
nd2_array = nd2.imread(nd2_path)
dark_400 = tiff.imread("images/20210531 Darkfield 400ms.tif").astype(np.float32)
dark_60 = tiff.imread("images/20210531 Darkfield 60ms.tif").astype(np.float32)

rfp = np.clip(nd2_array[:,0].astype(np.float32) - dark_400, 0, None) # substracting dark_frame
gfp = np.clip(nd2_array[:,1].astype(np.float32) - dark_60, 0, None)
bf = np.clip(nd2_array[:,2].astype(np.float32) - dark_60, 0, None)

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class SmallUNet(nn.Module):

    def __init__(self, in_channels=5, out_channels=2):

        super().__init__()

        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up3 = nn.ConvTranspose2d(
            256, 128, 2, stride=2
        )

        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(
            128, 64, 2, stride=2
        )

        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(
            64, 32, 2, stride=2
        )

        self.dec1 = DoubleConv(64, 32)

        self.output = nn.Conv2d(
            32,
            out_channels,
            1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        b = self.bottleneck(
            self.pool(e3)
        )

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.output(d1)


def dice_loss(pred, target, eps=1e-6):

    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum(dim=(1, 2))

    union = (
        pred.sum(dim=(1, 2))
        + target.sum(dim=(1, 2))
    )

    dice = (
        (2 * intersection + eps)
        / (union + eps)
    )

    return 1 - dice.mean()

def loss_function(pred, target):

    foreground_bce = F.binary_cross_entropy_with_logits(
        pred[:, 0],
        target[:, 0]
    )

    foreground_dice = dice_loss(
        pred[:, 0],
        target[:, 0]
    )

    foreground_loss = (
        foreground_bce
        + foreground_dice
    )

    # Convert center prediction to 0–1
    center_pred = torch.sigmoid(pred[:, 1])

    center_loss = F.mse_loss(
        center_pred,
        target[:, 1]
    )

    return foreground_loss + center_loss

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = SmallUNet().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

annotation_frames = [
    10,
    # 50,
    100,
    # 150,
    200,
    # 250,
    300,
    # 350,
    400,
    # 450
]

dataset = CellDataset(
    merged_ind_norm,
    annotation_frames,
    "cellpose_training/Cellsegmentation/annotations",
    patch_size=256,
    samples_per_frame=100
)

from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

for epoch in range(30):

    model.train()
    epoch_loss = 0

    print(f"\nStarting Epoch {epoch + 1}", flush=True)

    for batch_idx, (x, y) in enumerate(loader):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(x)

        loss = loss_function(pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/30 | "
            f"Batch {batch_idx + 1}/{len(loader)} | "
            f"Loss: {loss.item():.4f}",
            end="\r",
            flush=True
        )

    print(
        f"\nEpoch {epoch + 1}/30 finished | "
        f"Average loss: {epoch_loss / len(loader):.4f}",
        flush=True
    )

model.eval()

t = 10

x = get_temporal_window(
    merged_ind_norm,
    t,
    radius=2
)

x = torch.from_numpy(
    x[None]
).float().to(device)

with torch.no_grad():

    pred = model(x)

pred = torch.sigmoid(pred)

foreground = pred[0, 0].cpu().numpy()
centers = pred[0, 1].cpu().numpy()

cell_mask = foreground > 0.5

# find centers
coordinates = peak_local_max(
    centers,
    min_distance=20,
    threshold_abs=0.3
)

markers = np.zeros_like(
    cell_mask,
    dtype=np.int32
)

for i, (y, x) in enumerate(coordinates, start=1):

    markers[y, x] = i

distance = distance_transform_edt(
    cell_mask
)

labels = watershed(
    -distance,
    markers,
    mask=cell_mask
)

plt.figure(figsize=(8, 8))

# Original image
plt.imshow(
    merged_ind_norm[t],
    cmap="gray"
)

# Make background transparent
labels_masked = np.ma.masked_where(
    labels == 0,
    labels
)

# Overlay cells
plt.imshow(
    labels_masked,
    cmap='gist_ncar',
    alpha=0.5,
    interpolation="none"
)

plt.axis("off")
plt.show()