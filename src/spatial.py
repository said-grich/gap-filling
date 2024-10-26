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

class SpatialMethods():
    def __init__(self) -> None:
        pass
    
    
    def np_to_df(self,image):

        rows, cols = image.shape
        x_coords = np.arange(cols)
        y_coords = np.arange(rows)
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
        x_flat = x_mesh.ravel()
        y_flat = y_mesh.ravel()
        data_flat = image.ravel()
        result_array = np.column_stack((x_flat, y_flat, data_flat))
        df = pd.DataFrame(result_array, columns=['X', 'Y', 'Value'])

        return df
    
    