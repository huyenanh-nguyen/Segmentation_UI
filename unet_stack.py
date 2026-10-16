#!/usr/bin/env python3
"""Strictly stack only _instances TIFF frames from unet_predictions into multi-page TIFFs
and save them into unet_stack.
"""

from pathlib import Path
import numpy as np
import tifffile as tiff


def stack_unet_instances_strict() -> None:
    source_dir = Path("/u/hanguy/Segmentation_UI/unet_predictions")
    output_dir = Path("/u/hanguy/Segmentation_UI/unet_stack")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    subfolders = [item for item in source_dir.iterdir() if item.is_dir()]
    
    if not subfolders:
        print(f"No subfolders found inside '{source_dir}'.")
        return

    for subfolder in subfolders:
        # Strict filter: Must end with .tif / .tiff AND contain '_instances' in the name
        tiff_files = []
        for f in subfolder.iterdir():
            if f.is_file() and f.suffix.lower() in [".tif", ".tiff"] and "_instances" in f.name.lower():
                tiff_files.append(f)
        
        # Sort numerically by frame index extracted from the filename
        tiff_files = sorted(
            tiff_files,
            key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or 0)
        )
        
        if not tiff_files:
            print(f"Skipping folder '{subfolder.name}' (no instance files matched).")
            continue
        
        print(f"\nProcessing folder '{subfolder.name}' - Found {len(tiff_files)} instance files:")
        # Print a couple of matched files to verify they are strictly instances
        for tf in tiff_files[:3]:
            print(f"  - {tf.name}")
        if len(tiff_files) > 3:
            print(f"  - ... and {len(tiff_files) - 3} more.")

        # Read frames into a list and stack along the time axis (axis 0)
        frames = [tiff.imread(f) for f in tiff_files]
        stack = np.stack(frames, axis=0)  # Shape becomes (T, Y, X)
        
        output_path = output_dir / f"{subfolder.name}_instances_stack.tif"
        
        tiff.imwrite(
            output_path,
            stack,
            bigtiff=True,
            compression="zlib",
            photometric="minisblack",
            metadata={"axes": "TYX"}
        )
        print(f"-> Saved strict instance stack to: {output_path}")

    print(f"\nAll done! Stacks saved to: {output_dir}")


if __name__ == "__main__":
    stack_unet_instances_strict()