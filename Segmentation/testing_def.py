from pathlib import Path
import nd2
from anhs_watershed import Background
import matplotlib.pyplot as plt
import numpy as np


background = Background(darkframe_folder= str(Path.cwd()) + "/Data")

with open("access.txt", "r", encoding="utf-8") as file:
    content = [line.strip() for line in file]


datapath = list(Path(str(Path.cwd()) + content[1]).glob("*.nd2"))

for _, file in enumerate(datapath):
    name = str(file).split("/")[-1].split(".nd2")[0]
    mask_path = background.segment(
        nd2_path= str(file),
        channel_idx=0,
        median_filter_size=21,
        cell_diameter=50,
        output_folder= "Output",
        output_name=f"{name}_channel_0_mask_50um",
    )
