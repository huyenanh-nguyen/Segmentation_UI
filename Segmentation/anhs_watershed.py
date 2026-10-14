"""ND2 darkframe correction and frame-wise watershed segmentation.

The expected ND2 data layout is (time, channel, y, x), for example
(433, 3, 600, 600). The generated segmentation output is a TIFF stack
with layout (time, y, x).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2 as cv
import nd2
import numpy as np
import pandas as pd
from scipy import ndimage
import tifffile as tiff


class Background:
    """Process ND2 channels and produce frame-wise watershed masks."""

    DARKFIELD_FILENAMES = {
        50: "20201026 Darkfield 50ms binned.tif",
        60: "20210531 Darkfield 60ms.tif",
        100: "20201026 Darkfield 100ms binned.tif",
        200: "20201026 Darkfield 200ms binned.tif",
        300: "20220530 Darkfield 300 ms binned 003.tif",
        400: "20210531 Darkfield 400ms.tif",
    }

    EXPOSURE_RULES = {
        ("fakun", "gfp"): 60,
        ("fakun", "rfp"): 400,
        ("finn", "gfp"): 60,
        ("finn", "rfp"): 300,
        ("niranjan", "gfp"): 200,
        ("niranjan", "rfp"): 400,
    }

    def __init__(self, darkframe_folder: str | Path = "Data") -> None:
        """Create the processor.

        Args:
            darkframe_folder: Folder containing the six darkframe TIFF files.
                The default is the folder ``Data`` in the current directory.
        """
        self.darkframe_folder = Path(darkframe_folder)
        self._darkframe_cache: dict[int, np.ndarray] = {}

    def reading_channels(self, nd2_path: str | Path) -> dict[int, str]:
        """Read the channel names stored in an ND2 file."""
        channels: dict[int, str] = {}

        with nd2.ND2File(nd2_path) as nd2_file:
            for index, channel in enumerate(nd2_file.metadata.channels):
                channels[index] = channel.channel.name

        return channels

    def _load_darkframe(self, exposure_ms: int) -> np.ndarray:
        """Load and cache one 2D darkframe TIFF."""
        if exposure_ms not in self.DARKFIELD_FILENAMES:
            raise ValueError(f"No darkframe filename configured for {exposure_ms} ms.")

        if exposure_ms not in self._darkframe_cache:
            darkframe_path = (
                self.darkframe_folder / self.DARKFIELD_FILENAMES[exposure_ms]
            )

            if not darkframe_path.exists():
                raise FileNotFoundError(f"Darkframe not found: {darkframe_path}")

            darkframe = tiff.imread(darkframe_path).astype(np.float32)

            if darkframe.ndim != 2:
                raise ValueError(
                    f"Darkframe must be a 2D TIFF, but has shape {darkframe.shape}: "
                    f"{darkframe_path}"
                )

            self._darkframe_cache[exposure_ms] = darkframe

        return self._darkframe_cache[exposure_ms]

    @staticmethod
    def _parse_channel_name(channel_name: str) -> tuple[str | None, str | None]:
        """Extract laboratory user and fluorophore from a channel name."""
        clean_name = channel_name.lower().replace("_", " ").replace("-", " ")

        user: str | None = None
        if "finn" in clean_name:
            user = "finn"
        elif "fakun" in clean_name:
            user = "fakun"
        elif "niranjan" in clean_name or "nirjanjan" in clean_name:
            user = "niranjan"

        fluorophore: str | None = None
        if "gfp" in clean_name:
            fluorophore = "gfp"
        elif "rfp" in clean_name:
            fluorophore = "rfp"

        return user, fluorophore

    def darkfield_assignment(self, nd2_path: str | Path) -> dict[int, dict[str, Any]]:
        """Subtract the automatically selected darkframe from each known channel.

        The assignment uses the channel names embedded in the ND2 file. A channel
        name must contain both a known user (Finn, Fakun, Niranjan) and GFP or RFP.

        Returns:
            Mapping by channel index. ``raw_minus_dark`` has shape (T, Y, X), or is
            ``None`` if no exposure can be inferred from the channel name.
        """
        channels = self.reading_channels(nd2_path)
        nd2_array = nd2.imread(nd2_path).astype(np.float32)

        if nd2_array.ndim != 4:
            raise ValueError(
                "Expected an ND2 array with shape (time, channel, y, x), "
                f"but received {nd2_array.shape}."
            )

        assignments: dict[int, dict[str, Any]] = {}

        for channel_index, channel_name in channels.items():
            if channel_index >= nd2_array.shape[1]:
                raise IndexError(
                    f"Channel index {channel_index} is outside ND2 array shape "
                    f"{nd2_array.shape}."
                )

            user, fluorophore = self._parse_channel_name(channel_name)
            exposure_ms = self.EXPOSURE_RULES.get((user, fluorophore))

            raw_minus_dark: np.ndarray | None = None
            if exposure_ms is not None:
                darkframe = self._load_darkframe(exposure_ms)
                channel_stack = nd2_array[:, channel_index, :, :]

                if channel_stack.shape[1:] != darkframe.shape:
                    raise ValueError(
                        f"Channel {channel_index} has image size "
                        f"{channel_stack.shape[1:]}, but its darkframe for "
                        f"{exposure_ms} ms has shape {darkframe.shape}."
                    )

                raw_minus_dark = np.clip(channel_stack - darkframe[None, :, :], 0, None)

            assignments[channel_index] = {
                "channel_name": channel_name,
                "user": user,
                "fluorophore": fluorophore,
                "exposure_ms": exposure_ms,
                "raw_minus_dark": raw_minus_dark,
            }

        return assignments

    def median_blur_remove(
        self,
        nd2_path: str | Path,
        channel_idx: int,
        median_filter_size: int,
    ) -> np.ndarray:
        """Apply median background removal to each timepoint independently.

        The time axis is never filtered. For an input of (433, 600, 600), the
        median filter has the effective size (1, median_filter_size,
        median_filter_size).
        """
        if median_filter_size < 1:
            raise ValueError("median_filter_size must be at least 1.")

        assignments = self.darkfield_assignment(nd2_path)
        if channel_idx not in assignments:
            raise IndexError(f"Channel index {channel_idx} does not exist.")

        image_stack = assignments[channel_idx]["raw_minus_dark"]
        if image_stack is None:
            channel_name = assignments[channel_idx]["channel_name"]
            raise ValueError(
                f"No exposure/darkframe could be assigned to channel {channel_idx} "
                f"({channel_name!r})."
            )

        median_background = ndimage.median_filter(
            image_stack,
            size=(1, median_filter_size, median_filter_size),
        )

        return np.maximum(image_stack - median_background, 0)

    @staticmethod
    def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
        """Normalize a single 2D image to the range 0..255."""
        image = image.astype(np.float32, copy=False)
        minimum = float(image.min())
        maximum = float(image.max())

        if maximum <= minimum:
            return np.zeros(image.shape, dtype=np.uint8)

        normalized = (image - minimum) / (maximum - minimum)
        return (normalized * 255).astype(np.uint8)

    @staticmethod
    def _odd_kernel_size(size: int) -> int:
        """Return an OpenCV-compatible odd median-kernel size."""
        if size < 3:
            raise ValueError("cell_diameter must be at least 3 pixels.")
        return size if size % 2 else size + 1

    def _segment_frame(self, frame: np.ndarray, cell_diameter: int) -> np.ndarray:
        """Run the original-style preprocessing and watershed on one frame."""
        cell_diameter = self._odd_kernel_size(cell_diameter)
        frame_uint8 = self._normalize_to_uint8(frame)

        if not np.any(frame_uint8):
            return np.zeros(frame.shape, dtype=np.uint16)

        # Same principal operators as the original code: contrast multiplication,
        # broad blur, median filter and subtraction of the two images.
        contrast_img = np.clip(frame_uint8.astype(np.uint16) * 10, 0, 255).astype(
            np.uint8
        )

        blur_size = cell_diameter * 3
        blur_img = cv.blur(contrast_img, (blur_size, blur_size))
        median_img = cv.medianBlur(contrast_img, cell_diameter)
        subtracted_img = cv.subtract(median_img, blur_img)

        # Otsu foreground mask.
        _, foreground_mask = cv.threshold(
            subtracted_img,
            0,
            255,
            cv.THRESH_BINARY + cv.THRESH_OTSU,
        )

        morphology_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        foreground_mask = cv.morphologyEx(
            foreground_mask,
            cv.MORPH_OPEN,
            morphology_kernel,
            iterations=1,
        )
        foreground_mask = cv.morphologyEx(
            foreground_mask,
            cv.MORPH_CLOSE,
            morphology_kernel,
            iterations=2,
        )

        if not np.any(foreground_mask):
            return np.zeros(frame.shape, dtype=np.uint16)

        # Distance-transform peaks generate reliable watershed seeds.
        distance_img = cv.distanceTransform(foreground_mask, cv.DIST_L2, 5)
        distance_max = float(distance_img.max())
        if distance_max == 0:
            return np.zeros(frame.shape, dtype=np.uint16)

        _, sure_foreground = cv.threshold(
            distance_img,
            0.35 * distance_max,
            255,
            cv.THRESH_BINARY,
        )
        sure_foreground = sure_foreground.astype(np.uint8)

        sure_background = cv.dilate(
            foreground_mask,
            morphology_kernel,
            iterations=2,
        )
        unknown_region = cv.subtract(sure_background, sure_foreground)

        _, markers = cv.connectedComponents(sure_foreground)
        markers = markers.astype(np.int32)

        # OpenCV watershed convention: 1 is known background, 0 is unknown.
        markers += 1
        markers[unknown_region == 255] = 0

        watershed_input = cv.cvtColor(frame_uint8, cv.COLOR_GRAY2BGR)
        watershed_result = cv.watershed(watershed_input, markers)

        # -1 = watershed boundary, 1 = background, >1 = individual cells.
        labels = watershed_result.copy()
        labels[labels <= 1] = 0
        labels[labels > 1] -= 1

        return labels.astype(np.uint16)

    def segment(
        self,
        nd2_path: str | Path,
        channel_idx: int,
        median_filter_size: int,
        cell_diameter: int,
        output_folder: str | Path,
        output_name: str,
        tiff_compression_level: str | None = "zlib",
        ) -> Path:
        """Create and save a stacked watershed label mask for one ND2 channel.

        Args:
            nd2_path: Input ND2 file.
            channel_idx: Index of the ND2 channel to segment, e.g. 0.
            median_filter_size: 2D median background filter size in pixels.
            cell_diameter: Approximate cell diameter in pixels.
            output_folder: Destination folder; created if it does not exist.
            output_name: Desired output filename, with or without ``.tif``.
            tiff_compression_level: TIFF compression, for example ``"zlib"``.

        Returns:
            Path of the TIFF label stack with shape (T, Y, X). Background pixels
            have label 0; individual cells have positive labels.
        """
        output_directory = Path(output_folder)
        output_directory.mkdir(parents=True, exist_ok=True)

        output_filename = (
            output_name if output_name.lower().endswith((".tif", ".tiff"))
            else f"{output_name}.tif"
        )
        output_path = output_directory / output_filename

        image_stack = self.median_blur_remove(
            nd2_path=nd2_path,
            channel_idx=channel_idx,
            median_filter_size=median_filter_size,
        )

        if image_stack.ndim != 3:
            raise ValueError(
                "The selected channel must have shape (time, y, x), but got "
                f"{image_stack.shape}."
            )

        segmentation_stack = np.zeros(image_stack.shape, dtype=np.uint16)
        number_of_frames = image_stack.shape[0]

        for frame_index in range(number_of_frames):
            segmentation_stack[frame_index] = self._segment_frame(
                image_stack[frame_index],
                cell_diameter=cell_diameter,
            )

            if (frame_index + 1) % 25 == 0 or frame_index + 1 == number_of_frames:
                print(f"Segmented {frame_index + 1}/{number_of_frames} frames")

        tiff.imwrite(
            output_path,
            segmentation_stack,
            bigtiff=True,
            compression=tiff_compression_level,
            photometric="minisblack",
            metadata={
                "axes": "TYX",
                "description": (
                    "Frame-wise watershed segmentation mask. "
                    "0=background; positive values=cell labels."
                ),
            },
        )

        print(f"Segmentation TIFF stack saved to: {output_path}")
        return output_path

    def make_area_list(
        self,
        segmentation_tiff: str | Path,
        output_folder: str | Path,
        output_name: str = "cell_area.csv",
        min_cell_area: int = 0,
    ) -> Path:
        """Write cell areas and positions for every frame of a mask TIFF stack.

        Labels are frame-local: label 1 in one frame is not necessarily the same
        biological cell as label 1 in another frame unless tracking is added.
        """
        if min_cell_area < 0:
            raise ValueError("min_cell_area cannot be negative.")

        segmentation_stack = tiff.imread(segmentation_tiff)
        if segmentation_stack.ndim == 2:
            segmentation_stack = segmentation_stack[np.newaxis, :, :]
        if segmentation_stack.ndim != 3:
            raise ValueError(
                "Segmentation TIFF must have shape (time, y, x), but got "
                f"{segmentation_stack.shape}."
            )

        output_directory = Path(output_folder)
        output_directory.mkdir(parents=True, exist_ok=True)
        if not output_name.lower().endswith(".csv"):
            output_name = f"{output_name}.csv"
        output_path = output_directory / output_name

        rows: list[dict[str, float | int]] = []
        for frame_index, frame_mask in enumerate(segmentation_stack):
            cell_labels = np.unique(frame_mask)
            cell_labels = cell_labels[cell_labels > 0]

            for cell_label in cell_labels:
                cell_mask = frame_mask == cell_label
                area = int(np.count_nonzero(cell_mask))
                if area < min_cell_area:
                    continue

                y_coordinates, x_coordinates = np.where(cell_mask)
                rows.append(
                    {
                        "frame": frame_index,
                        "cell_label": int(cell_label),
                        "area_px": area,
                        "min_x": int(x_coordinates.min()),
                        "min_y": int(y_coordinates.min()),
                        "max_x": int(x_coordinates.max()),
                        "max_y": int(y_coordinates.max()),
                        "center_x": float(x_coordinates.mean()),
                        "center_y": float(y_coordinates.mean()),
                    }
                )

        columns = [
            "frame",
            "cell_label",
            "area_px",
            "min_x",
            "min_y",
            "max_x",
            "max_y",
            "center_x",
            "center_y",
        ]
        pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)

        print(f"Area CSV saved to: {output_path}")
        return output_path

    def make_substacks(
        self,
        segmentation_tiff: str | Path,
        image_tiff: str | Path,
        output_folder: str | Path,
        min_cell_area: int = 0,
        padding: int = 5,
        tiff_compression_level: str | None = "zlib",
    ) -> Path:
        """Save one masked TIFF crop for each cell in each frame.

        ``image_tiff`` must be an intensity TIFF stack with exactly the same shape
        as the segmentation stack. The output is arranged in folders such as
        ``frame_0000/cell_001.tif``.
        """
        if min_cell_area < 0:
            raise ValueError("min_cell_area cannot be negative.")
        if padding < 0:
            raise ValueError("padding cannot be negative.")

        segmentation_stack = tiff.imread(segmentation_tiff)
        image_stack = tiff.imread(image_tiff)

        if segmentation_stack.ndim == 2:
            segmentation_stack = segmentation_stack[np.newaxis, :, :]
        if image_stack.ndim == 2:
            image_stack = image_stack[np.newaxis, :, :]

        if segmentation_stack.shape != image_stack.shape:
            raise ValueError(
                "Segmentation and image stacks must have the same shape. "
                f"Mask: {segmentation_stack.shape}; image: {image_stack.shape}."
            )

        output_directory = Path(output_folder)
        output_directory.mkdir(parents=True, exist_ok=True)

        _, image_height, image_width = segmentation_stack.shape
        for frame_index, (frame_mask, frame_image) in enumerate(
            zip(segmentation_stack, image_stack, strict=True)
        ):
            frame_directory = output_directory / f"frame_{frame_index:04d}"
            frame_directory.mkdir(parents=True, exist_ok=True)

            cell_labels = np.unique(frame_mask)
            cell_labels = cell_labels[cell_labels > 0]

            for cell_label in cell_labels:
                cell_mask = frame_mask == cell_label
                area = int(np.count_nonzero(cell_mask))
                if area < min_cell_area:
                    continue

                y_coordinates, x_coordinates = np.where(cell_mask)
                min_y = max(0, int(y_coordinates.min()) - padding)
                max_y = min(image_height, int(y_coordinates.max()) + padding + 1)
                min_x = max(0, int(x_coordinates.min()) - padding)
                max_x = min(image_width, int(x_coordinates.max()) + padding + 1)

                cropped_image = frame_image[min_y:max_y, min_x:max_x]
                cropped_mask = cell_mask[min_y:max_y, min_x:max_x]
                cropped_cell = cropped_image * cropped_mask

                cell_path = frame_directory / f"cell_{int(cell_label):03d}.tif"
                tiff.imwrite(
                    cell_path,
                    cropped_cell,
                    compression=tiff_compression_level,
                    photometric="minisblack",
                )

            if (frame_index + 1) % 25 == 0 or frame_index + 1 == len(segmentation_stack):
                print(f"Created crops for {frame_index + 1}/{len(segmentation_stack)} frames")

        print(f"Cell crops saved to: {output_directory}")
        return output_directory



