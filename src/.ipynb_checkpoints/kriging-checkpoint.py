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
class krigingGapFilling:
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
    
    
    
    def spatial_kriging(self,cloudy_image):
    
        try:
                df=self.np_to_df(cloudy_image)

                Xgrid=np.sort(df["X"].unique())
                Ygrid=np.sort(df["Y"].unique())
                df.dropna(inplace=True)
                X = df[['X', 'Y']].values
                y = df['Value'].values

                del df
                gc.collect()

                # Estimate the variogram
                bin_center, gamma = gs.vario_estimate_unstructured((X[:,0], X[:,1]), y)

                #we use three variogram models : Gaussain, Exponential and Spherical
                models = {
                    "Gaussian": gs.Gaussian(dim=2, var=1, len_scale=10, nugget=0),
                    "Exponential": gs.Exponential(dim=2, var=1, len_scale=10, nugget=0),
                    "Spherical": gs.Spherical(dim=2, var=1, len_scale=10, nugget=0.5),
                }


                params = {}
                scores = {}
                best_model = ''
                best_rmse = np.inf

                for model in models:

                        fit_model = models[model]

                        # Fit the variogram model
                        para, pcov = fit_model.fit_variogram(bin_center, gamma, max_eval=1000000)        

                        y_pred = fit_model.variogram(bin_center)
                        rmse = np.sqrt(mean_squared_error(gamma, y_pred))

                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_model = model

                        params[model] = para
                        scores[model] = rmse    

                        

                param_dict = params[best_model]
                sill = param_dict['var']
                range_ = param_dict['len_scale']
                nugget = param_dict['nugget']

                del bin_center, gamma, params, scores, models, fit_model, para, pcov, y_pred, rmse
                gc.collect()

                # Prepare variogram model parameters for PyKrige
                pykrige_params = {'sill': sill, 'range': range_, 'nugget': nugget}




                # Use the best model and parameters in OrdinaryKriging
                OK = OrdinaryKriging(
                    X[:, 0],
                    X[:, 1],
                    y,
                    variogram_model=best_model.lower(),
                    variogram_parameters=pykrige_params,  # Provide the parameters
                    verbose=False,
                    enable_plotting=False
                )

                filled_image, ss = OK.execute("grid", Xgrid, Ygrid)


                del X, y, Xgrid, Ygrid, OK, pykrige_params
                gc.collect()


                return filled_image

        except MemoryError:
                print(" ----------------------------------- memory error ")
                return None

        except Exception as e:
                print(f"-----------------------------------An error occurred: {e}")
                return None
        