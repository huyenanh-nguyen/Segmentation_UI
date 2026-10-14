from pathlib import Path
import nd2
from anhs_watershed import Background
import matplotlib.pyplot as plt
import numpy as np

file_path = Path.cwd() / "Data" / "20250912 7nM 212-1F5_TAB3_MyD88 1um_est100mol.nd2"


# with nd2.ND2File(str(file_path)) as f:
#     # 1. Access channel information
#     for i, channel in enumerate(f.metadata.channels):
#         print(f"Channel {i}: {channel.channel.name}")

# # [testing code]

image = Path.cwd() / "Data" / "20250912 7nM 212-1F5_TAB3_MyD88 1um_est100mol.nd2"


# Example: change the three paths and the segmentation settings as needed.
background = Background(darkframe_folder= str(Path.cwd()) + "/Data")

mask_path = background.segment(
    nd2_path= str(image),
    channel_idx=0,
    median_filter_size=21,
    cell_diameter=21,
    output_folder= "Output",
    output_name="channel_0_mask",
)

# Optional: write per-frame cell areas and positions.
background.make_area_list(
    segmentation_tiff=mask_path,
    output_folder="Output",
    output_name="channel_0_cell_area",
    min_cell_area=21**2,
)

# Optional: requires an intensity TIFF stack with shape (T, Y, X).
# background.make_substacks(
#     segmentation_tiff=mask_path,
#     image_tiff="Output/channel_0_intensity.tif",
#     output_folder="Output/channel_0_cells",
#     min_cell_area=21**2,
#     padding=5,
# )







# plt.imshow(Background().median_blur_remove(image, 1,10), vmax = np.max(Background().median_blur_remove(image, 1,10))*0.5)
# plt.savefig("test.png", bbox_inches="tight")
# plt.close()

# Background().segment(image, 0, 10,10,50, "watershed_0_idx")






# from pathlib import Path
# import nd2

# file_path = Path.cwd() / "Data" / "20221216 3nM 231-A8_NEMO_TRAF6_1um_38mol 001.nd2"

# with nd2.ND2File(str(file_path)) as f:
#     # 1. Inspect general channel info
#     print("--- Channel Metadata ---")
#     for i, ch in enumerate(f.metadata.channels):
#         name = ch.channel.name
#         print(f"Channel {i}: {name}")
#         # Depending on the file version, channel settings might be stored here:
#         # print(ch)

#     # 2. Inspect frame-level metadata (where actual active exposure times per frame/channel are stored)
#     print("\n--- Frame-level Metadata & Exposure ---")
#     num_frames = f.sizes.get('T', f.attributes.sequenceCount)
    
#     for idx in range(min(5, num_frames)):
#         meta = f.frame_metadata(idx)
#         print(f"Frame {idx} metadata:", meta)
        
#         # If you want to check if exposure time is explicitly in the frame metadata structure:
#         # (You can inspect the printed output of `meta` to see fields like exposure time)


# from pathlib import Path
# from nd2reader import ND2Reader

# file_path = Path.cwd() / "Data" / "20221216 3nM 231-A8_NEMO_TRAF6_1um_38mol 001.nd2"

# with ND2Reader(str(file_path)) as images:
#     # Print available metadata keys for the first frame
#     print(images[0].metadata)
    
#     # Often exposes attributes like exposure time directly in milliseconds or seconds
#     # e.g., images[0].metadata.get('t_ms') or similar keys depending on the file version


# from pathlib import Path
# from aicsimageio import AICSImage

# file_path = Path.cwd() / "Data" / "20221216 3nM 231-A8_NEMO_TRAF6_1um_38mol 001.nd2"

# img = AICSImage(file_path)

# # View metadata using the embedded OME XML object
# print(img.metadata)