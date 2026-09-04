import numpy as np
import tifffile as tiff
import nd2

class Segementation:


    def __init__(self, image: str):
        """
        Args:
            image: Path to the ND2 image.
        """
        self.image = image
        self.data = None

    def convert_nd2_to_array(self):
        """
        Convert ND2 image to NumPy array.

        Returns:
            np.ndarray: ND2 image data.
        """
        self.data = nd2.imread(self.image)

        print("ND2 array shape:", self.data.shape)

        return self.data

    def substracting_darkframe(self, darkframe: str, channel: int):
        """
        Subtract a dark frame from one channel.

        Args:
            darkframe: Path to dark-frame TIFF.
            channel: Channel index.

        Returns:
            np.ndarray: Dark-frame corrected image.
        """

        if self.data is None:
            self.convert_nd2_to_array()

        darkframe_array = tiff.imread(darkframe)

        # Expected ND2 shape:
        # (time, channels, y, x)
        image = self.data[:, channel].astype(np.float32)

        substract = np.clip(
            image - darkframe_array,
            0,
            None
        )

        return substract

