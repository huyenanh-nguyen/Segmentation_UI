import math, tifffile, time
import numpy as np
from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage

class Background:
    def remove_frame(self, image_path, frame_path):
        