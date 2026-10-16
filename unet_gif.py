#!/usr/bin/env python3
"""Convert multi-page TIFF stacks of instance masks into animated GIFs
and save them into a new folder named unet_gif.
"""

from pathlib import Path
import numpy as np
import tifffile as tiff
from PIL import Image


def convert_tiff_stacks_to_gifs() -> None:
    # Define exact absolute paths
    source_dir = Path("/u/hanguy/Segmentation_UI/unet_stack")
    output_dir = Path("/u/hanguy/Segmentation_UI/unet_gif")
    
    # Create the output directory if it doesn't already exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    # Find all multi-page TIFF stacks in the source directory
    tiff_files = sorted(list(source_dir.glob("*.tif")) + list(source_dir.glob("*.tiff")))
    
    if not tiff_files:
        print(f"No TIFF stack files found inside '{source_dir}'.")
        return

    for tiff_file in tiff_files:
        print(f"Processing '{tiff_file.name}'...")
        
        # Read the multi-page TIFF stack (shape: T, Y, X)
        stack = tiff.imread(tiff_file)
        
        if stack.ndim != 3:
            print(f"Skipping '{tiff_file.name}': Expected a 3D stack (T, Y, X), but got shape {stack.shape}.")
            continue
        
        frames = []
        for frame in stack:
            # Normalize instance values or categorical labels to 8-bit (0-255) for valid GIF visualization
            min_val = frame.min()
            max_val = frame.max()
            
            if max_val > min_val:
                normalized = ((frame - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                normalized = np.zeros(frame.shape, dtype=np.uint8)
            
            # Convert NumPy array to a Pillow image
            pil_img = Image.fromarray(normalized)
            frames.append(pil_img)
        
        if not frames:
            print(f"Skipping '{tiff_file.name}': No valid frames extracted.")
            continue
        
        # Define output GIF file path inside 'unet_gif'
        output_path = output_dir / f"{tiff_file.stem}.gif"
        
        # Save as an animated GIF (duration=100ms per frame, loop=0 for infinite loop)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0
        )
        print(f"-> Saved animated GIF to: {output_path}")

    print(f"\nAll GIFs successfully created and saved to: {output_dir}")


if __name__ == "__main__":
    convert_tiff_stacks_to_gifs()