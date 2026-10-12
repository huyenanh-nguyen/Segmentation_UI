# from pathlib import Path
# import nd2

# file_path = Path.cwd() / "Data" / "20250912 7nM 212-1F5_TAB3_MyD88 1um_est100mol.nd2"

# with nd2.ND2File(str(file_path)) as f:
#     # 1. Access channel information
#     for i, channel in enumerate(f.metadata.channels):
#         print(f"Channel {i}: {channel.channel.name}")

#     # 2. Access frame metadata safely using f.sizes or total sequence count
#     num_frames = f.sizes.get('T', f.attributes.sequenceCount)
    
#     for idx in range(min(5, num_frames)):
#         meta = f.frame_metadata(idx)
#         print(f"Frame {idx}:", meta)

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


from pathlib import Path
from aicsimageio import AICSImage

file_path = Path.cwd() / "Data" / "20221216 3nM 231-A8_NEMO_TRAF6_1um_38mol 001.nd2"

img = AICSImage(file_path)

# View metadata using the embedded OME XML object
print(img.metadata)