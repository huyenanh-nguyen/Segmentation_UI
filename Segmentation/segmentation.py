from torch.utils.data import Dataset
import torch
import math, tifffile, os, time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob
import tifffile as tiff
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
from pathlib import Path
import nd2
from ipywidgets import interact, IntSlider, widgets
from ipywidgets import interactive_output
from scipy.ndimage import center_of_mass
from scipy.ndimage import gaussian_filter


class Segementation:

    def __init__(self, image : str):
        """
        Args:
            image (str): Path of image
        """
        self.image = image

    def convert_nd2_to_array(self):
        """
        Returns:
            Array: converting nd2 images to array
        """
        nd2_array = nd2.imread(self.image)
        return nd2_array

    def substracting_darkframe(self, darkframe : str, channel : int):
        """_summary_

        Args:
            darkframe (str): path
            channel (int): index of channel (depending on how many channels was taken)

        Returns:
            Array: array without the camera noises
        """
        image = self.convert_nd2_to_array()
        darkframe = tiff.imread(darkframe)

        substract = np.clip(image[:,int].astype(np.float32) - darkframe, 0, None)

        return substract

