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
warnings.filterwarnings('ignore')
import random
import seaborn as sns
import copy
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
import concurrent.futures
from tqdm import tqdm

class SpatioTemporalGapFilling:
    def __init__(self) -> None:
        pass
    
    def Stack_to_Datarame(self,stack_images):
    
        """
        Description: The purpose of this function is to convert a dictionary representing the stack_images into a DataFrame indexed
                        by dates of acquisition
        Args:    -stack_images: the initial stack of image stocked in a dictionnary where the keys are the acquisition dates, and 
                                values are images of the corresponding dates
        Returns: -training_ts_x : Dataframe containing all the pixels time series and indexed by dates of acquisition

        """
            
        data_frames = []

        for timestamp, array in stack_images.items():
            # Reshape the 2D array to a 1D array (flatten it)
            flattened_array = array.flatten()
            # Create a DataFrame with the flattened array as a single column
            df = pd.DataFrame({timestamp: flattened_array})
            data_frames.append(df)

        # Concatenate the DataFrames along columns
        training_ts_x = pd.concat(data_frames, axis=1)

        # Transpose the DataFrame to have datetime dates as the index
        training_ts_x = training_ts_x.T
        training_ts_x = training_ts_x.sort_index()

        return training_ts_x
    
    def coefficient_matrix(self,dates, avg_days_yr=365.25, num_coefficients=8):
        
        """
        Description : Create the linear harmonic model used for preprocessing the overlap region
        
        Args :   -'dates' : the acquisition dates of the images
        
        Return : -'matrix' : the harmonic model for each of the acquisition dates
        """

        w = 2 * np.pi / avg_days_yr
        matrix = np.zeros(shape=(len(dates), 7), order='F')

        cos = np.cos
        sin = np.sin

        w12 = w * dates
    
        
        matrix[:, 0] = dates
        matrix[:, 1] = cos(w12)
        matrix[:, 2] = sin(w12)

        if num_coefficients >= 6:
            w34 = 2 * w12
            matrix[:, 3] = cos(w34)
            matrix[:, 4] = sin(w34)

        if num_coefficients >= 8:
            w56 = 3 * w12
            matrix[:, 5] = cos(w56)
            matrix[:, 6] = sin(w56)

        return matrix
    
    
    def lasso_fill(self,dates, X, avg_days_yr=365.25):
        
        """
        Description : the pixel time series in overlap regions also have cloudcontaminated data, those training data need to be
                        preprocessed. this function aims at filling gaps for each of these pixel time series:
                            * fit a linear harmonic model with the pixel time serie (only for known values) 
                            * fill the gaps on the pixel time serie using the created model (predicting unknonw values)
        Return : -'X': the prepocessed dataframe 
        """
        
        coef_matrix = self.coefficient_matrix(dates, avg_days_yr) #(n_feature, 7)   #coef_matrix.shape : (76, 7)
        lasso = linear_model.Lasso(alpha=1)
    
        X_invalid = np.isnan(X)
        X_valid = ~X_invalid

        for i_row in range(X.shape[0]):
            if len(coef_matrix[X_invalid[i_row, :], :])==0:
                continue
            model = lasso.fit(coef_matrix[X_valid[i_row, :], :], X[i_row, :][X_valid[i_row, :]])
            X[i_row, :][X_invalid[i_row, :]] = model.predict(coef_matrix[X_invalid[i_row, :], :])

        return X
    
    def find_lastValley(self,arr_1d):
    
        '''
        Description: this function allows retrieving the local minimun in an univariate array "arr_1d", it is used for the finding
                    the overlap thresold for selecting the overlap region pixels
                    
        returns : local minimuns of the array
        '''
        
        # https://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
        min_idx = (np.diff(np.sign(np.diff(arr_1d))) > 0).nonzero()[0] + 1

        if (len(min_idx) == 0): # if there is only one peak
            return 21 # at least 21 clear obs for the harmonic model
        else:
            return np.argmin(abs(arr_1d - arr_1d[min_idx][-1]))


    
    def save_cluster(self,ts, n_clusters=20, n_cpu=-1, method='KMean'):
        
        """
        Description: Cluster the training data "ts" using Kmeans algorithm into 'n_clusters' cluster
        
        Args: - ts : dataframe to be clustered
            - n_clusters: the number of clusters to create
            - n_cpu : number of cpu to be used, it impacts the speed of the algorithm, generally it is set to '-1' to be able to 
                        use all the disponible cpu 
            - method : present the clustering method, by default we use the KMeans algorithm          
        
        returns : -'cls': the KMeans model
        """
        
        if method == 'KMean':
            cls = KMeans(n_clusters)
            labels = cls.fit_predict(ts)
            return cls
        else:
            print('Not implemented!')
            return None




    
    
    def sampling_strata(self,ts_y, centroids, labels, sample_size=100):
        
        '''
        Description: using the controids and lables of the training data, extract samples of size "sample_size" depending on the 
                    similarity to the target pixel time serie 'ts_y' and return Index of selected samples
                    
        args:   ts_y: Pixel time series to be filled
                centroids: Center time series of each cluster classes.
                labels: labels of clusters
                sample_size : the size of the selected sample
        '''
        
        tmp_numpy=ts_y.to_numpy()[np.newaxis, :]
        
        weight = 1/self.cluster_obs_dis(tmp_numpy, centroids)
        
        weight = weight / np.nanmax(weight)
        #if ts_y is at one of the centroids
        weight_idx = np.isfinite(weight)
        # weight[~weight_idx] = 2*np.nanmax(weight)
        weight[~weight_idx] = 1.0

        prob = np.take(weight, labels)
        if np.sum(prob) > 0:
            prob = prob / np.sum(prob)
            idx = np.random.choice(np.asarray(range(len(labels))), sample_size, replace=False, p=prob)
            return idx
        else:
            print('diag: ts_y, weight_idx, weight, prob', ts_y, weight_idx, weight, prob)
            print('total probability equal to 0!')
            return None

    def gap_fill_pixel(self,ts_y, ts_x, cluster_model, iter_num=10, sample_size=100):
    
        '''
        Description :  Predicting the  gaps in the target pixel 'ts_y', the filling steps are:
                        - repeat 'iter_num' times:
                            1- Stratify random samples of 'sample_size' pixels from the training data 'ts_x' depending to the 
                               similarity to the target pixel 
                            2- predict the target pixel gaps using Lasso regression between the target pixel 'ts_y' and the training 
                               sample
                        - fill the pixel gaps using the median of the predictions 
                        
        Args :   -'ts_y': the target pixel 
                 -'ts_x': training data, which represent the overlap region pixels values
                 
               
        Return : -'impute_y': target pixel imputed                  
        '''
        
        labels = cluster_model.labels_
        centroids = cluster_model.cluster_centers_
    
        impute_y = np.zeros_like(ts_y, dtype=float)
        if ts_y.isna().sum()==0 : #if no missing data
            return ts_y
        elif ts_y.count()==0 :     #if no valid data
            return impute_y

        
    
        ts_temp = []
        for i_ter in range(iter_num):
            sample_idx = self.sampling_strata(ts_y, centroids, labels, sample_size=sample_size)
            ts_temp.append(self.lasso_impute(ts_y.copy(), ts_x.iloc[:, sample_idx], replace=False))
            
        
        ts_temp = np.array(ts_temp)
        impute_y = np.nanmedian(ts_temp, axis=0).astype(float)
                
        impute_y=pd.DataFrame(impute_y,  index=ts_x.index , columns=['Value'])
        
        return impute_y



    def lasso_impute(self,ts_y, ts_x, replace=False):
        
        '''
        Description: fitting a Lasso regression model between "ts_y" and "ts_x", and predict the missing values
        
        Args :   -'ts_y': the target pixel 
                 -'ts_x': selected training data
        
        returns : -"ts_y": target pixel with missing values replaced with conserving the clear ones if 'replace' is False
        '''
    
        y_valid_idx = ts_y.notna()
        
        # Create a DataFrame containing only valid samples from ts_x and ts_y
        valid_data = ts_x[y_valid_idx].copy()
        valid_data['ts_y'] = ts_y[y_valid_idx]
        
        # Split valid_data into X and y
        X = valid_data.drop(columns=['ts_y'])
        y = valid_data['ts_y']
        
        # Initialize the Lasso model
        cls = Lasso()
        
        if len(y) > 0:
            model = cls.fit(X, y)
            
            # Use the trained model to predict missing values in ts_y
            if replace:
                ts_y[y_valid_idx] = model.predict(ts_x[y_valid_idx])
            else:
                ts_y[~y_valid_idx] = model.predict(ts_x[~y_valid_idx])
        
        return ts_y



    def cluster_obs_dis(self,X, centroids):
        """
        Description: This function calculate distances from time series X to centroids using only valid observations, this distances 
                    represents the metric of similarity 


        Args:  X:  Pixel time series to be filled
            centroids: Center time series of each cluster classes
            
        returns: the calculated distance

        """
        
        dis = np.zeros((X.shape[0], centroids.shape[0]))
        invalid = np.isnan(X)

        for i_cls in range(centroids.shape[0]):
            dif = X - np.tile(centroids[i_cls, :], X.shape[0]).reshape(X.shape)
            dif[invalid] = 0        
            dis[:, i_cls] = np.sqrt(np.sum(dif**2, axis=1)/(centroids.shape[1]-np.sum(invalid, axis=1)))

        return dis
    
    #this three functions are used for constructing the dataframe 'ts_x' which represents the overlap region
    def gather_training(self,training_ts_x, min_number_of_clear_obs=21):
        
        """
        Description : - extract the overlap region (pixels time series presenting the higher density of clear values) using 
                        the maximun between the threshold that divides two peaks and the 'min_number_of_clear_obs', the region 
                        presents the training data
                    - Cluster the training data using Kmeans into 20 cluster
        
        Args: training_ts_x: Dataframe containing all the pixels time series and indexed by dates of acquisition
        
        Return : - training_data: Dataframe containing selected pixels time series values
                - cluster_model: the cluster model of 20 cluster created with Kmeans algorithm
                - overlap_thesh: the overlap threshold for selecting a pixel or not in the overlap region
                
        """

        hist, bins = np.histogram(training_ts_x.count(), bins=range(np.max(training_ts_x.count())))
        overlap_thesh = bins[self.find_lastValley(hist)]
        
        # vmin_number_of_clear_obs is set to 21, becaause it usually requires at least 21 clear observations in the training data
        overlap_thesh = np.max([overlap_thesh, min_number_of_clear_obs]) 

        # Create a new DataFrame with selected columns
        training_ts_x = training_ts_x[training_ts_x.columns[training_ts_x.count() > overlap_thesh]]
        acq_datelist = training_ts_x.index.values
        reference_date = np.datetime64('1970-01-01')
        acq_datelist = (acq_datelist - reference_date).astype('timedelta64[D]').astype(float)

        training_data_fill = self.lasso_fill(acq_datelist, training_ts_x.values.T)
        training_data = pd.DataFrame(training_data_fill.T, index=training_ts_x.index.values ,columns=training_ts_x.columns )

        cluster_model=self.save_cluster(training_data.values.T)
                        
        return training_data,cluster_model,overlap_thesh
    def retrieve_TS(self,stack_images, overlap_thesh):
        """
        Description: depending of the found overlap threshold value "overlap_thesh", we stock the pixels time series of the overlap
                    region in the dictionnary 'dict_TS_overlap_region', the pixels to be filled are stocked in 'dict_TS_y'
                    
        Returns : 'dict_TS_overlap_region': dictionnary of overlap pixels, the keys of the dictionnary are the pixel indices and 
                                        the values are the corresponding time serie in a format of Dataframe
                'dict_TS_y': dictionnary of same structure as 'dict_TS_overlap_region', but dedicated to pixels not in overlap 
                            region
        """
        
        dict_TS_y = {}
        dict_TS_overlap_region={}
        
        temporary_image=next(iter(stack_images.values()))
        indices = [(i, j) for i in range(temporary_image.shape[0]) for j in range(temporary_image.shape[1])]
        
        for index in indices:
            
            TS={key:stack_images[key][tuple(index)] for key in stack_images}
            TS=pd.DataFrame.from_dict(TS, orient='index', columns=['Value'])
            TS=TS.sort_index()
            
            if TS.count().sum()<=overlap_thesh:
                dict_TS_y[tuple(index)]=TS
            
            else:
                dict_TS_overlap_region[tuple(index)]=TS
        
        return dict_TS_overlap_region,dict_TS_y
    
    def dict_of_indices_To_stack_image(self,dict_TS,stack_images):
    
        """
        Description: function to transfer the dictionnary of filled pixels time series into a stack of image
        
        Args: - dict_TS : dictionnary containing all the pixels time series of the image filled 
            - stack_images: the initial stack of image stocked in a dictionnary where the keys are the acquisition dates, and 
                            values are images of the corresponding dates
            
        Return: -filled_stack_images: dictionnary containing the filled images
        """

        temp_arr=next(iter(stack_images.values()))
        temp_zeros_image = np.zeros((temp_arr.shape[0], temp_arr.shape[1]))
        temp_df=next(iter(dict_TS.values()))


        filled_stack_images = {date: np.copy(temp_zeros_image) for date in stack_images.keys()}

        # Iterate over dict_TS and fill dict_img_stack
        for index in dict_TS:
                ts_y = dict_TS[tuple(index)]["Value"]
                for date in ts_y.index:
                    filled_stack_images[date][tuple(index)] = ts_y.loc[date]
                    
        return filled_stack_images
    def process_item(self, args):
        index, ts_x, cluster_model, local_dict_TS_y = args
        ts_y = local_dict_TS_y[tuple(index)]["Value"]
        impute_y = self.gap_fill_pixel(ts_y, ts_x, cluster_model, iter_num=10, sample_size=100)
        return tuple(index), impute_y
        
    def reconstruct_filled_stack_images(self,stack_images,min_number_of_clear_obs=21):
        """
        Decsription: the function covers all the filling process for an inital stack of images
        Args: - stack_images: the initial stack of image stocked in a dictionnary where the keys are the acquisition dates, and 
                            values are images of the corresponding dates
            - min_number_of_clear_obs : the minimant number of clear values for a pixel time series to be took in consideration
                                        for the overlap region
        Returns: -filled_stack_images : dictionnary containing the filled images
        
        """
        
        training_ts_x=self.Stack_to_Datarame(stack_images)
        

        ts_x,cluster_model,overlap_thesh=self.gather_training(training_ts_x,min_number_of_clear_obs)
        
        
        print(f'overlap_thesh : {overlap_thesh}')
        start_time = time.time()

        dict_TS_overlap_region,dict_TS_y=self.retrieve_TS(stack_images, overlap_thesh)
        
        for index in dict_TS_overlap_region:
            ts_x_target=dict_TS_overlap_region[tuple(index)]["Value"]
            if ts_x_target.isna().sum()>0:

                acq_datelist = ts_x_target.index.values
                reference_date = np.datetime64('1970-01-01')
                acq_datelist = (acq_datelist - reference_date).astype('timedelta64[D]').astype(float)
                TS_target = ts_x_target.to_numpy().reshape(1, -1)
                training_data_fill=self.lasso_fill(acq_datelist, TS_target)
                ts_x_filled=pd.DataFrame(training_data_fill.T, index=ts_x_target.index.values, columns=['Value'])
                dict_TS_overlap_region[tuple(index)]=ts_x_filled

        
        for index in dict_TS_y:
            ts_y=dict_TS_y[tuple(index)]["Value"]
            impute_y=self.gap_fill_pixel(ts_y, ts_x, cluster_model, iter_num=5, sample_size=50)
            dict_TS_y[tuple(index)]=impute_y

        
        #combine 'dict_TS_overlap_region' and "dict_TS_y" into one single dictionnary
        dict_TS_y.update(dict_TS_overlap_region)
        
        #reconstruct the stack of image using the dictionnary of filled pixel time series
        filled_stack_images=self.dict_of_indices_To_stack_image(dict_TS_y,stack_images)
        # plot_images_with_dates(filled_stack_images)
        elapsed_time = time.time() - start_time
        print(f"The function took {elapsed_time} seconds to execute.")


        return filled_stack_images  
            