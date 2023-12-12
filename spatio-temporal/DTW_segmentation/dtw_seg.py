import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import random
import re
import json
from datetime import datetime
import pickle
import os
import warnings
import copy
from pykrige.ok3d import OrdinaryKriging3D
from dtaidistance import dtw
import cv2
import time
import concurrent.futures
from tqdm import tqdm
warnings.filterwarnings('ignore')


class DTWSegmentation:
    def __init__(self,images_shape,max_no_growth_iterations,min_growth_size) -> None:
        self.images_shape=images_shape
        self.max_no_growth_iterations = max_no_growth_iterations  # Set a threshold for the number of iterations without significant growth
        self.min_growth_size = min_growth_size    
    
    
    
    #Determine Similarity Threshold 
    def calculate_similarity_threshold_mean(self,image_series):
        return np.array([np.nanmean(image_array) for image_array in image_series ])
    
    def calculate_similarity_threshold_std(self,image_series):
        return np.array([np.nanmean(image_array) for image_array in image_series ])

    #Determine the Number and Location of the Seeds
    def is_pixel_valid_across_time_series(self,pixel, image_series):
        """Check if a pixel is not NaN in any image of the time series."""
        x, y = pixel
        return not any(np.isnan(image_series[i][x, y]) for i in range(len(image_series)))

    def find_edges(self,image):
        
        # Apply Sobel filters for edge detection
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        
        # Combine the two edges
        combined_edge = cv2.magnitude(sobelx, sobely)
        
        return combined_edge

    def select_seeds_from_edges_time_series(self,image_series, num_seeds=10, min_distance=20):
        seeds_per_image=[]
        for image_ in image_series:
            edge_image = self.find_edges(image_)# Edge detection on the first image, for example
        
        
            # Flatten the edge image and sort by edge intensity
            flat_indices = np.argsort(edge_image.ravel())[::-1]  # Descending order

            seeds = []
            for flat_index in flat_indices:
                if len(seeds) >= num_seeds:
                    break

                seed_candidate = np.unravel_index(flat_index, edge_image.shape)

                # Check if seed is valid across the time series and not too close to existing seeds
                if self.is_pixel_valid_across_time_series(seed_candidate, image_series) and \
                all(np.linalg.norm(np.array(seed_candidate) - np.array(existing_seed)) >= min_distance for existing_seed in seeds):
                    seeds.append(seed_candidate)
            seeds_per_image.append(seeds)

        return seeds_per_image
    
    
    #Identifies the neighboring pixels
    def find_neighbors(self,pixel, image_shape):
        """
        Find the 8-connected neighbors of a given pixel in an image.

        :param pixel: A tuple (x, y) representing the pixel coordinates.
        :param image_shape: The shape of the image (height, width).
        :return: A set of tuples representing the coordinates of the neighboring pixels.
        """
        x, y = pixel
        neighbors = set()

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                # Skip the center pixel itself
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy

                # Check if neighbor is within the image boundaries
                if 0 <= nx < image_shape[0] and 0 <= ny < image_shape[1]:
                    neighbors.add((nx, ny))

        return neighbors
    
    #Compute DTW Distance Between the Time Series of the Seeds and Their Neighbors
    def compute_dtw_distances(self,image_time_series, seed, neighbors):
    
        # Extract time-series data for the seed
        seed_time_series = image_time_series[:, seed[0], seed[1]]
        # Compute DTW distance with each neighbor
        distances = {}
        for neighbor in neighbors:
            neighbor_time_series = image_time_series[:, neighbor[0], neighbor[1]]
            distance = dtw.distance(seed_time_series, neighbor_time_series)
            distances[tuple(neighbor)] = distance
        return distances
    
    #Continue Examining All Neighbors Until No Similar Neighbor is Found
    def grow_region(self,image_time_series, seed, threshold):
        image_shape=self.images_shape
        initial_neighbors = self.find_neighbors(seed, image_shape)
        region = set([tuple(seed)])
        candidates = set(initial_neighbors)

        # Early stopping parameters
        no_growth_counter = 0
        max_no_growth_iterations = self.max_no_growth_iterations  # Set a threshold for the number of iterations without significant growth
        min_growth_size = self.min_growth_size        # Set a minimum number of pixels to consider as significant growth

        last_region_size = len(region)

        while candidates:
            current = candidates.pop()

            # Check if the current pixel is a NaN in any of the time series images
            is_nan = np.any(np.isnan(image_time_series[:, current[0], current[1]]))
            # Only proceed if the current pixel is not a NaN
            if not is_nan:
                distances = self.compute_dtw_distances(image_time_series, seed, [current])
    #             print(f"distances  ---> {distances}")
                if distances[tuple(current)] < threshold:
                    region.add(tuple(current))
                    new_neighbors = self.find_neighbors(current, image_shape)
                    candidates.update(new_neighbors)

            # Check for significant growth
            if len(region) - last_region_size >= min_growth_size:
                no_growth_counter = 0  # Reset counter if significant growth occurred
                last_region_size = len(region)
            else:
                no_growth_counter += 1  # Increment counter if no significant growth

            # Early stopping if there's been no significant growth for max_no_growth_iterations
            if no_growth_counter >= max_no_growth_iterations:
                break

        return region

    


    def run_segmentation_process(self,image_series, nan_mask, seeds_per_image, thresholds):
        # Initialize the DTW Segmentation class

        results_regions = []
        for image_index in tqdm(range(len(image_series)), desc="Processing Images"):
            if nan_mask[image_index]:
                image_regions = []

                seeds_of_image = seeds_per_image[image_index]
                for seed in tqdm(seeds_of_image, desc=f"Image {image_index} Seeds", leave=False):
                    result = self.grow_region(image_series, seed, thresholds[image_index])
                    image_regions.append(result)

                results_regions.append(image_regions)
        return results_regions

