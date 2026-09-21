from pathlib import Path
import zipfile
from zipfile import BadZipFile
import numpy as np
import tifffile as tiff
from read_roi import read_roi_zip, read_roi_file
import ijroi
from roifile import ImagejRoi

import cv2
import numpy as np


def add_roi_to_mask(roi, mask_array, label):
    """Rasterizes an ImageJ ROI object onto the mask_array with a specific label value."""
    try:
        coords = None

        # Call methods with () if they are callable methods/functions
        if hasattr(roi, "integer_coordinates"):
            if callable(roi.integer_coordinates):
                coords = roi.integer_coordinates()
            else:
                coords = roi.integer_coordinates
        elif hasattr(roi, "coordinates"):
            if callable(roi.coordinates):
                coords = roi.coordinates()
            else:
                coords = roi.coordinates

        if coords is None or len(coords) == 0:
            return False

        coords = np.array(coords, dtype=np.int32)

        # Apply position offset if present in the ROI object
        x_off = getattr(roi, "left", 0)
        y_off = getattr(roi, "top", 0)

        # Standard shape check: ensure Nx2 array of (X, Y) points
        if coords.ndim == 2 and coords.shape[1] == 2:
            coords[:, 0] += x_off
            coords[:, 1] += y_off

            pts = coords.reshape((-1, 1, 2))
            cv2.fillPoly(mask_array, [pts], color=int(label))
            return True

        return False

    except Exception as e:
        print(f"Error drawing ROI to mask: {e}")
        return False

class Trainingpreperation:

    def process_rois_to_tiff(
        self, annotationpath, outputfolder, height, width, suffix="", process_zips=True, process_rois=True
    ):
        """
        Converts ROI files (standalone .roi or inside .zip archives) into TIFF mask images.
        Accepts either a directory or a direct path to a .zip / .roi file.
        """
        input_path = Path(annotationpath)
        out_dir = Path(str(outputfolder) + "/" + str(suffix))
        out_dir.mkdir(parents=True, exist_ok=True)

        # Handle both single files and directory inputs
        if input_path.is_file():
            zip_files = [input_path] if input_path.suffix.lower() == ".zip" and process_zips else []
            roi_files = [input_path] if input_path.suffix.lower() == ".roi" and process_rois else []
        else:
            zip_files = list(input_path.glob("*.zip")) if process_zips else []
            roi_files = list(input_path.glob("*.roi")) if process_rois else []

        # 1. Process ZIP archives
        for zip_path in zip_files:
            print(f"\nProcessing ZIP: {zip_path.name}")

            try:
                with zipfile.ZipFile(zip_path) as z:
                    roi_names = [
                        name for name in z.namelist() if name.lower().endswith(".roi")
                    ]
            except BadZipFile:
                print(f"  Skipping {zip_path.name}: not a valid ZIP file")
                continue

            print(f"  Found {len(roi_names)} ROIs inside {zip_path.name}")
            if not roi_names:
                print("  No ROI files inside ZIP, skipping.")
                continue

            mask_array = np.zeros((height, width), dtype=np.uint16)

            with zipfile.ZipFile(zip_path) as z:
                label = 1
                for roi_name in roi_names:
                    try:
                        with z.open(roi_name) as f:
                            roi = ImagejRoi.frombytes(f.read())

                        success = add_roi_to_mask(roi, mask_array, label)
                        if success:
                            label += 1
                        else:
                            print(f"  Skipping ROI: {roi_name}")
                    except Exception as e:
                        print(f"  Error reading {roi_name}: {e}")

            output_mask = out_dir / f"{zip_path.stem.replace('.roi', '')}.tif"
            tiff.imwrite(output_mask, mask_array)

            print(f"  Saved mask: {output_mask}")
            print(f"  Labels present: {np.unique(mask_array)}")

        # 2. Process standalone .roi files
        for roi_path in roi_files:
            print(f"\nProcessing ROI: {roi_path.name}")
            mask_array = np.zeros((height, width), dtype=np.uint16)

            try:
                roi = ImagejRoi.fromfile(roi_path)
                success = add_roi_to_mask(roi, mask_array, label=1)

                if not success:
                    print("  Could not extract coordinates, skipping.")
                    continue

                output_mask = out_dir / f"{roi_path.stem}.tif"
                tiff.imwrite(output_mask, mask_array)

                print(f"  Saved mask: {output_mask}")
                print(f"  Labels present: {np.unique(mask_array)}")

            except Exception as e:
                print(f"  Error processing {roi_path.name}: {e}")

        return None