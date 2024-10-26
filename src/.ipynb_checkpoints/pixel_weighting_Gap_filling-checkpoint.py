from multiprocessing import Pool
import numpy as np
import time
import warnings
import copy
from dtaidistance import dtw
import time
import numpy as np
from multiprocessing import Pool ,cpu_count

warnings.filterwarnings('ignore')


def process_pixel(pixel_info, images_time_series, region_mean, nan_mask, gap_filling_model):
    px, image_mean_region = pixel_info
    similar_regions = gap_filling_model.find_most_similar_region(px, images_time_series, image_mean_region, nan_mask)
    if similar_regions is not None:
        px_value = gap_filling_model.predict_gap_value_with_weighted_average(similar_regions)
        return px, px_value
    return px, None

class PixelWeightingGapFilling:
    def __init__(self) -> None:
        self.nan_count=0
        
    def extract_time_series_values(self,pixel, image_series ,nan_mask):
        values = []
        for index, image in enumerate(image_series):
            if not nan_mask[index]:
                values.append(image[pixel])
        return values

    def precompute_region_means(self,image_series, regions,nan_mask):
        # Initialize a dictionary to store the time series of means for each region
        region_means_time_series = {region_index: [] for region_index in range(len(regions))}

        for index,image in enumerate(image_series):
            if(nan_mask[index]):
                
                for region_index, region in enumerate(regions):
                    pixel_sum = 0
                    pixel_count = 0
                    
                    # Iterate over each pixel in the region
                    for (row, col) in list(region):

                        pixel_value = image[row, col]
                        
                        # Only consider valid pixel values (ignore NaNs)
                        if not np.isnan(pixel_value):
                            pixel_sum += pixel_value
                            pixel_count += 1
                    
                    # Calculate the mean value if there are valid pixels
                    if pixel_count > 0:
                        mean_value = pixel_sum / pixel_count
                    else:
                        mean_value = np.nan  # Handle cases where all pixel values are NaN
                    
                    # Append the mean value to the corresponding time series
                    region_means_time_series[region_index].append(mean_value)


        return region_means_time_series
    
    def find_most_similar_region(self,gap_pixel, image_series, regions_mean_time_series,nan_mask):
        pixel_values_in_non_gap_images = self.extract_time_series_values(gap_pixel, image_series ,nan_mask)
        if np.isnan(pixel_values_in_non_gap_images).any():
                return None
        regions_dtw=[]
        
        for key in regions_mean_time_series.keys():
            dtw_ds = dtw.distance(regions_mean_time_series[key],pixel_values_in_non_gap_images)
            regions_dtw.append({"dtw":dtw_ds,"region_index":key ,"region_non_gaped_images":np.array(regions_mean_time_series[key]),"pixel_values_in_non_gap_images":np.array(pixel_values_in_non_gap_images)})
        sorted_regions_dtw = sorted(regions_dtw, key=lambda x: x['dtw'])
        return sorted_regions_dtw[:10]
    
    
    def predict_gap_value_with_weighted_average(self,similar_regions):
        # Extract DTW distances and region means into separate arrays
        dtw_distances = np.array([region['dtw'] for region in similar_regions])
        region_means = np.array([np.mean(region['region_non_gaped_images']) for region in similar_regions])

        # Compute weights
        weights = 1 / (dtw_distances )  # Vectorized operation

        # Check if total weight is zero
        if np.sum(weights) == 0:
            return None

        # Compute weighted average
        weighted_average = np.sum(weights * region_means) / np.sum(weights)
        return weighted_average
    

    
    
    
    def process_pixel(self,px, images_time_series, image_mean_region, nan_mask):
        similar_regions = self.find_most_similar_region(px, images_time_series, image_mean_region, nan_mask)
        if similar_regions is not None:
            px_value = self.predict_gap_value_with_weighted_average(similar_regions)
            return px, px_value
        return px, None

    def process_image(self, image_index, image, images_time_series, segmentation_result, nan_mask):
        if nan_mask[image_index]:
           
            selected_image = copy.deepcopy(image)
            image_mean_region = self.precompute_region_means(images_time_series, list(segmentation_result[self.nan_count]), nan_mask)
            nan_indices = np.where(np.isnan(selected_image))
            pixels = list(zip(nan_indices[0], nan_indices[1]))

            with Pool(30) as pool:
                pixel_values = pool.starmap(self.process_pixel, [(px, images_time_series, image_mean_region, nan_mask) for px in pixels])

            for px, px_value in pixel_values:
                if px_value is not None:
                    selected_image[px[0]][px[1]] = px_value
            self.nan_count+=1
            return selected_image
        else:
            return image

    def fill_images(self, segmentation_result, images_time_series, nan_mask):
        filled_images=[]
        for index, image in enumerate(images_time_series):
            self.nan_count=0
            filled_images.append(self.process_image(index, image, images_time_series, segmentation_result, nan_mask))
        return filled_images
