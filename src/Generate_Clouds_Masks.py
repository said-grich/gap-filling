import numpy as np
from multiprocessing import Pool
import geopandas as gpd
import multiprocessing
import random
import pickle
class MasksGenerator():
    def __init__(self) -> None:
        pass
    def cloud_mask_circle(self,image, percentage):
        '''
        Description: This function simulates clouds over an image in the shape of a single circular cloud centered at the center 
                    of the image.
                    
        Args:
            - 'image':      A 2D NumPy array representing the input image on which clouds will be simulated.
            - 'percentage': A value indicating the percentage of total cloud cover over the image. It should verify the 
                            condition 0 <= missing_percentage <= 100.
        
        Return:
            - 'image_with_clouds': A 2D NumPy array representing the image with one artificial circular cloud centered at the center
                                    of image where the size depends on the specified percentage.         
        '''        
        y_size, x_size = image.shape
        xx, yy = np.meshgrid(np.arange(x_size), np.arange(y_size))

        size = int(np.sqrt((percentage/100) * x_size * y_size / np.pi))
        circle_mask = (xx - x_size/2)**2 + (yy - y_size/2)**2 <= size**2

        image = image.astype(np.float32)
        
        return circle_mask
    def generate_cloud_pixels(self,center, cloud_size, border_margin, height, width):
        """
        Generate pixels for a single cloud.
        """
        cloud_pixels = set([center])

        while len(cloud_pixels) < cloud_size:
            # Choose a random pixel on the border of the cloud
            border_pixel = random.choice(list(cloud_pixels))
            neighbors = [
                (border_pixel[0] + dy, border_pixel[1] + dx)
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            ]
            random.shuffle(neighbors)

            for neighbor in neighbors:
                y, x = neighbor
                if (border_margin <= y < height - border_margin and
                    border_margin <= x < width - border_margin):
                    cloud_pixels.add(neighbor)
                    break
                    # Early break once a new cloud pixel is added

        return cloud_pixels
    
    def cloud_mask_separated(self,image, missing_percentage, num_clouds=3, border_margin=5):
    
        """
        Description: This function generates a cloud mask with separated cloud clusters, introducing missing values (NaN) to 
                    represent cloudy values, allowing for the simulation of cloud cover with customizable parameters.

        Args:

            - 'image':              2D NumPy array representing the input image on which clouds will be simulated.
            - 'missing_percentage': Value indicating the percentage of total cloud cover over the image. It should verify the
                                    condition 0 <= missing_percentage <= 100.
            - 'num_clouds':         Integer representing the number of distinct clouds to simulate.
            - 'border_margin':      Integer specifying the number of pixel margins to keep as clear from clouds.
        
        Return:
            - 'image_with_clouds':  2D NumPy array representing the image with artificial clouds based on the specified parameters.
            
        """
        
        if missing_percentage < 0 or missing_percentage > 100:
            raise ValueError("Missing percentage should be between 0 and 100")
        if num_clouds < 1:
            raise ValueError("Number of clouds should be a positive integer")

        height, width = image.shape
        total_pixels = height * width
        missing_pixels = int(total_pixels * missing_percentage / 100)

        # Select center points for each cloud without overlap
        cloud_centers = []
        while len(cloud_centers) < num_clouds:
            center = (random.randint(border_margin, height - border_margin - 1),
                    random.randint(border_margin, width - border_margin - 1))
            if all(np.linalg.norm(np.array(center) - np.array(other_center)) >= border_margin * 2
                for other_center in cloud_centers):
                cloud_centers.append(center)

        # Distribute the missing pixels among clouds
        cloud_sizes = np.random.multinomial(missing_pixels, [1/num_clouds] * num_clouds)

        # Create a mask with the same shape as the input image
        mask = np.zeros((height, width), dtype=bool)

        cloud_args = [(center, size, border_margin, height, width) 
                    for center, size in zip(cloud_centers, cloud_sizes)]

        # Use multiprocessing to generate cloud pixels
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            cloud_pixels_list = pool.starmap(self.generate_cloud_pixels, cloud_args)

        # Update the mask with cloud pixels from all clouds
        for cloud_pixels in cloud_pixels_list:
            for y, x in cloud_pixels:
                mask[y, x] = True

        # Apply the mask to the input image
        # image_with_clouds = np.where(mask, np.nan, image)

        return mask


if __name__ == "__main__":
    
    cloud_percentages = [10,20, 30, 40, 50]
    num_clouds_list = [1,2,3,4,5]
    image_size =(182, 161)   # Example image size, adjust as needed

    # Create an example image (could be any image data you are working with)
    example_image = np.ones(image_size)
    masks_dic={}

    masks_dic["separted"]={}
    masks_dic["circle"]=[]
    
    masksGenerator=MasksGenerator()
    
    

    for percentage in cloud_percentages:
            print(f"-Circle ------> {percentage} \n")
            mask_circle = masksGenerator.cloud_mask_circle(example_image, percentage,)
            masks_dic["circle"].append(mask_circle)


    for num_clouds_ in num_clouds_list:
        masks_dic["separted"][num_clouds_]=[]
        for percentage in cloud_percentages:
            print(f"-Circle ------> {percentage}  -- {num_clouds_} \n")
            mask_separated = masksGenerator.cloud_mask_separated(example_image, percentage ,num_clouds=num_clouds_)
            masks_dic["separted"][num_clouds_].append(mask_separated)
    
    file_path = './data/masks_final_11_12.pkl'

    # Saving the dictionary as a pickle file
    with open(file_path, 'wb') as file:
        pickle.dump(masks_dic, file)