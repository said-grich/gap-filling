import numpy as np
import os
import time
import cv2
import random
import warnings
import gstools as gs
import os
import psutil
import threading
import time
import gc
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import datetime
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from pykrige.ok import OrdinaryKriging
from pykrige.rk import Krige
from scipy.spatial import distance_matrix

# Ignore warnings
warnings.filterwarnings('ignore')

class InterpolationGapFilling():
    def __init__(self) -> None:
        pass
    
    
    def interpolate_spatial(self,cloudy_image,method):

        # Get the coordinates of the known and missing pixels
        known_coords = np.argwhere(~np.isnan(cloudy_image))
        missing_coords = np.argwhere(np.isnan(cloudy_image))

        # Get the values of the known pixels
        known_values = cloudy_image[~np.isnan(cloudy_image)]

        # Interpolate the missing pixel values using the selected method: nearest, linear, cubic.....
        missing_values = griddata(
            known_coords, known_values, missing_coords, method=method
        )

        # Fill in the missing pixel values in the original image
        filled_image = cloudy_image.copy()
        for (i, j), value in zip(missing_coords, missing_values):
            filled_image[i, j] = value

        return filled_image