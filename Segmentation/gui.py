import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from segmentation import Segementation

import numpy as np
import tifffile as tiff


class ND2Viewer(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("ND2 Segmentation")
        self.geometry("900x950")

        # --------------------------------
        # Store segmentation object
        # --------------------------------

        self.segmentation = None

        # Currently displayed Tk image
        self.current_photo = None

        # --------------------------------
        # Intensity window
        # --------------------------------

        self.vmin = 0
        self.vmax = 255

        # --------------------------------
        # Dark frame storage
        # --------------------------------

        # Dictionary:
        #
        # channel number -> dark frame path
        #
        self.darkframes = {}

        # Dictionary:
        #
        # channel number -> BooleanVar
        #
        self.darkframe_enabled = {}

        # Dictionary:
        #
        # channel number -> button
        #
        self.darkframe_buttons = {}

        # --------------------------------
        # ROW 1: Upload button
        # --------------------------------

        self.upload_button = tk.Button(
            self,
            text="Upload ND2 File",
            command=self.load_nd2,
            font=("Arial", 14)
        )

        self.upload_button.pack(pady=10)

        # --------------------------------
        # File path
        # --------------------------------

        self.path_label = tk.Label(
            self,
            text="No file selected",
            wraplength=800
        )

        self.path_label.pack(pady=5)

        # --------------------------------
        # Image display
        # --------------------------------

        self.image_label = tk.Label(
            self,
            text="No image loaded"
        )

        self.image_label.pack(
            expand=True,
            padx=10,
            pady=10
        )

        # --------------------------------
        # Mouse wheel controls
        # --------------------------------

        # Windows / Linux
        self.image_label.bind(
            "<MouseWheel>",
            self.mouse_wheel
        )

        # Mac
        self.image_label.bind(
            "<Button-4>",
            self.scroll_up
        )

        self.image_label.bind(
            "<Button-5>",
            self.scroll_down
        )

        # --------------------------------
        # Channel slider
        # --------------------------------

        self.channel_label = tk.Label(
            self,
            text="Channel: 0"
        )

        self.channel_label.pack()

        self.channel_slider = tk.Scale(
            self,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            length=600,
            command=self.update_image
        )

        self.channel_slider.pack(pady=5)

        # --------------------------------
        # Time slider
        # --------------------------------

        self.time_label = tk.Label(
            self,
            text="Time: 0"
        )

        self.time_label.pack()

        self.time_slider = tk.Scale(
            self,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            length=600,
            command=self.update_image
        )

        self.time_slider.pack(pady=5)

        # --------------------------------
        # Intensity window information
        # --------------------------------

        self.window_label = tk.Label(
            self,
            text="Window: vmin = 0 | vmax = 255"
        )

        self.window_label.pack(pady=5)

        # --------------------------------
        # DARK FRAME SECTION
        # --------------------------------

        self.darkframe_frame = tk.LabelFrame(
            self,
            text="Dark frame correction",
            padx=10,
            pady=10
        )

        self.darkframe_frame.pack(
            fill="x",
            padx=20,
            pady=10
        )


    # ====================================
    # LOAD ND2
    # ====================================

    def load_nd2(self):

        file_path = filedialog.askopenfilename(
            title="Select ND2 file",
            filetypes=[
                ("ND2 files", "*.nd2"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        # --------------------------------
        # Save path
        # --------------------------------

        self.path_label.config(
            text=file_path
        )

        print("Selected ND2:")
        print(file_path)

        # --------------------------------
        # Create segmentation object
        # --------------------------------

        self.segmentation = Segementation(file_path)

        # --------------------------------
        # Convert ND2 → NumPy
        # --------------------------------

        data = self.segmentation.convert_nd2_to_array()

        print("Array shape:", data.shape)

        # --------------------------------
        # Check dimensions
        # --------------------------------

        if data.ndim != 4:

            raise ValueError(
                f"Expected 4D array "
                f"(time, channels, y, x), "
                f"but got {data.shape}"
            )

        # --------------------------------
        # Get dimensions
        # --------------------------------

        time_length = data.shape[0]
        channel_length = data.shape[1]

        print(f"Time points: {time_length}")
        print(f"Channels: {channel_length}")

        # --------------------------------
        # Configure channel slider
        # --------------------------------

        self.channel_slider.config(
            from_=0,
            to=channel_length - 1
        )

        # --------------------------------
        # Configure time slider
        # --------------------------------

        self.time_slider.config(
            from_=0,
            to=time_length - 1
        )

        # Reset sliders
        self.channel_slider.set(0)
        self.time_slider.set(0)

        # --------------------------------
        # Create dark frame controls
        # --------------------------------

        self.create_darkframe_controls(
            channel_length
        )

        # --------------------------------
        # Initial intensity window
        # --------------------------------

        first_image = data[0, 0].astype(
            np.float32
        )

        self.vmin = float(
            first_image.min()
        )

        self.vmax = float(
            first_image.max()
        )

        print(
            f"Initial intensity window: "
            f"vmin={self.vmin}, "
            f"vmax={self.vmax}"
        )

        # --------------------------------
        # Display first image
        # --------------------------------

        self.show_image()


    # ====================================
    # CREATE DARK FRAME CONTROLS
    # ====================================

    def create_darkframe_controls(
        self,
        channel_length
    ):

        # --------------------------------
        # Remove old controls
        # --------------------------------

        for widget in self.darkframe_frame.winfo_children():
            widget.destroy()

        # Reset dictionaries
        self.darkframes = {}
        self.darkframe_enabled = {}
        self.darkframe_buttons = {}

        # --------------------------------
        # Create one row per channel
        # --------------------------------

        for channel in range(channel_length):

            row = tk.Frame(
                self.darkframe_frame
            )

            row.pack(
                fill="x",
                pady=2
            )

            # Channel label
            channel_label = tk.Label(
                row,
                text=f"Channel {channel}:",
                width=12,
                anchor="w"
            )

            channel_label.pack(
                side="left"
            )

            # --------------------------------
            # Upload button
            # --------------------------------

            button = tk.Button(
                row,
                text="Upload darkframe",
                command=lambda c=channel:
                    self.upload_darkframe(c)
            )

            button.pack(
                side="left",
                padx=5
            )

            self.darkframe_buttons[channel] = button

            # --------------------------------
            # Checkbox
            # --------------------------------

            enabled = tk.BooleanVar(
                value=False
            )

            checkbox = tk.Checkbutton(
                row,
                text="Use",
                variable=enabled,
                command=self.update_image
            )

            checkbox.pack(
                side="left",
                padx=5
            )

            self.darkframe_enabled[channel] = enabled


    # ====================================
    # UPLOAD DARK FRAME
    # ====================================

    def upload_darkframe(self, channel):

        file_path = filedialog.askopenfilename(
            title=f"Select dark frame for Channel {channel}",
            filetypes=[
                ("TIFF files", "*.tif *.tiff"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        # --------------------------------
        # Read dark frame
        # --------------------------------

        darkframe = tiff.imread(
            file_path
        )

        print(
            f"Dark frame for channel {channel}:"
        )

        print(file_path)

        print(
            "Dark frame shape:",
            darkframe.shape
        )

        # --------------------------------
        # Check dark frame dimensions
        # --------------------------------

        data = self.segmentation.data

        expected_shape = data.shape[-2:]

        if darkframe.shape != expected_shape:

            raise ValueError(
                f"Dark frame shape {darkframe.shape} "
                f"does not match image shape "
                f"{expected_shape}"
            )

        # --------------------------------
        # Store path
        # --------------------------------

        self.darkframes[channel] = file_path

        # --------------------------------
        # Update button text
        # --------------------------------

        self.darkframe_buttons[channel].config(
            text="Darkframe loaded"
        )

        # --------------------------------
        # Automatically enable it
        # --------------------------------

        self.darkframe_enabled[channel].set(
            True
        )

        # --------------------------------
        # Display corrected image
        # --------------------------------

        self.show_image()


    # ====================================
    # UPDATE IMAGE
    # ====================================

    def update_image(self, value=None):

        if self.segmentation is None:
            return

        self.show_image()


    # ====================================
    # MOUSE WHEEL
    # ====================================

    def mouse_wheel(self, event):

        if self.segmentation is None:
            return

        if event.delta > 0:
            self.scroll_up(event)
        else:
            self.scroll_down(event)


    # ====================================
    # SCROLL UP
    # ====================================

    def scroll_up(self, event=None):

        if self.segmentation is None:
            return

        self.change_vmax(+1)


    # ====================================
    # SCROLL DOWN
    # ====================================

    def scroll_down(self, event=None):

        if self.segmentation is None:
            return

        self.change_vmax(-1)


    # ====================================
    # CHANGE VMAX
    # ====================================

    def change_vmax(self, direction):

        # --------------------------------
        # Get current image
        # --------------------------------

        image = self.get_current_image()

        image_min = float(
            image.min()
        )

        image_max = float(
            image.max()
        )

        # --------------------------------
        # Determine step
        # --------------------------------

        intensity_range = (
            image_max - image_min
        )

        step = max(
            intensity_range * 0.01,
            1
        )

        # --------------------------------
        # Change vmax
        # --------------------------------

        self.vmax += direction * step

        # --------------------------------
        # Keep valid
        # --------------------------------

        self.vmax = max(
            self.vmin + 1,
            self.vmax
        )

        self.vmax = min(
            image_max,
            self.vmax
        )

        # --------------------------------
        # Update
        # --------------------------------

        self.show_image()


    # ====================================
    # GET CURRENT IMAGE
    # ====================================

    def get_current_image(self):

        data = self.segmentation.data

        time_index = self.time_slider.get()
        channel_index = self.channel_slider.get()

        # --------------------------------
        # Extract image
        # --------------------------------

        image = data[
            time_index,
            channel_index
        ].astype(np.float32)

        # --------------------------------
        # Dark frame correction
        # --------------------------------

        if (
            channel_index in self.darkframes
            and self.darkframe_enabled[
                channel_index
            ].get()
        ):

            darkframe_path = (
                self.darkframes[channel_index]
            )

            darkframe = tiff.imread(
                darkframe_path
            ).astype(np.float32)

            image = np.clip(
                image - darkframe,
                0,
                None
            )

        return image


    # ====================================
    # DISPLAY IMAGE
    # ====================================

    def show_image(self):

        if self.segmentation is None:
            return

        # --------------------------------
        # Current indices
        # --------------------------------

        time_index = self.time_slider.get()
        channel_index = self.channel_slider.get()

        # --------------------------------
        # Get image
        # --------------------------------

        image = self.get_current_image()

        # --------------------------------
        # Update labels
        # --------------------------------

        self.time_label.config(
            text=f"Time: {time_index}"
        )

        self.channel_label.config(
            text=f"Channel: {channel_index}"
        )

        # --------------------------------
        # Make sure window is valid
        # --------------------------------

        if self.vmax <= self.vmin:

            self.vmax = self.vmin + 1

        # --------------------------------
        # Apply intensity window
        # --------------------------------

        image = np.clip(
            image,
            self.vmin,
            self.vmax
        )

        # --------------------------------
        # Convert to 0–255
        # --------------------------------

        image = (
            (image - self.vmin)
            / (self.vmax - self.vmin)
            * 255
        )

        image = image.astype(
            np.uint8
        )

        # --------------------------------
        # NumPy → PIL
        # --------------------------------

        pil_image = Image.fromarray(
            image
        )

        # Resize while maintaining aspect ratio
        pil_image.thumbnail(
            (700, 500)
        )

        # --------------------------------
        # PIL → Tkinter
        # --------------------------------

        self.current_photo = ImageTk.PhotoImage(
            pil_image
        )

        self.image_label.config(
            image=self.current_photo
        )

        # --------------------------------
        # Window information
        # --------------------------------

        darkframe_status = ""

        if (
            channel_index in self.darkframes
            and self.darkframe_enabled[
                channel_index
            ].get()
        ):
            darkframe_status = (
                " | Darkframe: ON"
            )
        else:
            darkframe_status = (
                " | Darkframe: OFF"
            )

        self.window_label.config(
            text=(
                f"Window: "
                f"vmin = {self.vmin:.1f} | "
                f"vmax = {self.vmax:.1f}"
                f"{darkframe_status}"
            )
        )

