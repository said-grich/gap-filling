
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

class IdwGapFilling:
    def __init__(self) -> None:
        pass
    def spatial_idw(self,cloudy_image, power=1, percentage=0.1):
    
        Y, X = np.mgrid[0:cloudy_image.shape[0], 0:cloudy_image.shape[1]]
        coords = np.column_stack((Y.flatten(), X.flatten()))

        # Split the coordinates into known and unknown points
        known_mask = np.isfinite(cloudy_image.flatten())
        unknown_mask = ~known_mask
        sample_points = coords[known_mask]
        unknown_points = coords[unknown_mask]
        
        # Get the known values
        values = cloudy_image.flatten()[known_mask]

        # Calculate num_points as a percentage of total points
        num_points = int(percentage * len(values))
        
        
        # Perform IDW interpolation
        interpolated_values = self.idw_calculs(sample_points, unknown_points, values, num_points, power=power)

        # Create a copy of the original image
        filled_image = cloudy_image.copy()

        # Replace the NaN values in the image with the interpolated values
        filled_image[unknown_points[:,0], unknown_points[:,1]] = interpolated_values

        
        # Deleting not required variables to free memory
        del Y, X, coords, known_mask, unknown_mask, sample_points, unknown_points, values, num_points, interpolated_values
        gc.collect()

        return filled_image

    def idw_calculs(self,sample_points, unknown_points, values, num_points, power=2):
    
        # Calculate the distance matrix between known and unknown points
        distances = distance_matrix(sample_points, unknown_points)
        distances = np.round(distances, 4)

        # Set the zero distances to a very small number to prevent division by zero
        distances[distances == 0] = 1e-10

        # Get indices of the 'num_points' nearest known points for each unknown point
        nearest_indices = np.argpartition(distances, num_points, axis=0)[:num_points]

        # Sort indices of each partition
        for i in range(nearest_indices.shape[1]):
            nearest_indices[:,i] = nearest_indices[:,i][np.argsort(distances[nearest_indices[:,i], i])]

        # Create an array of column indices the same shape as nearest_indices
        column_indices = np.arange(nearest_indices.shape[1])

        # Create new arrays for the nearest known points and their distances
        nearest_points = sample_points[nearest_indices, :]
        nearest_distances = distances[nearest_indices, column_indices]
        nearest_values = values[nearest_indices]

        # Calculate weights using the inverse of the distance raised to the power
        weights = 1 / np.power(nearest_distances, power)

        # Reshape arrays for correct broadcasting in the multiplication
        weights = weights[..., np.newaxis]
        nearest_values = nearest_values[..., np.newaxis]

        # Calculate the interpolated values
        interpolated_values = np.sum(weights * nearest_values, axis=0) / np.sum(weights, axis=0)
        
        # Deleting not required variables to free memory
        del distances, nearest_indices, column_indices, nearest_points, nearest_distances, nearest_values, weights
        gc.collect()

        return interpolated_values.squeeze()  # remove singleton dimensions