import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import copy
import time

class SSRF_GapFilling:
    def __init__(self, cloudy_image, auxiliary_bands_images):
        self.cloudy_image = cloudy_image
        self.auxiliary_bands_images = auxiliary_bands_images
        self.number_of_bands = len(auxiliary_bands_images)

    def generate_df(self):
        columns_train = ["{}_val{}".format(band, i) for band in self.auxiliary_bands_images.keys() for i in range(1, 10)]
        columns_test = columns_train.copy()

        columns_train.append("target_value")
        columns_test.append("location i")
        columns_test.append("location j")

        df_SS_train = pd.DataFrame(columns=columns_train)
        df_SS_test = pd.DataFrame(columns=columns_test)

        for i in range(1, self.cloudy_image.shape[0] - 1):
            for j in range(1, self.cloudy_image.shape[1] - 1):
                array = np.empty((0))
                if not np.isnan(self.cloudy_image[i, j]):
                    for band_auxiliary_image in self.auxiliary_bands_images.values():
                        patch = band_auxiliary_image[i - 1:i + 2, j - 1:j + 2].ravel()
                        array = np.append(array, patch)
                    array = np.append(array, self.cloudy_image[i, j])
                    array = array.reshape((1, self.number_of_bands * 9 + 1))
                    df_SS_train = pd.concat([df_SS_train, pd.DataFrame(array, columns=columns_train)], ignore_index=True)
                else:
                    for band_auxiliary_image in self.auxiliary_bands_images.values():
                        patch = band_auxiliary_image[i - 1:i + 2, j - 1:j + 2].ravel()
                        array = np.append(array, patch)
                    array = np.append(array, i)
                    array = np.append(array, j)
                    array = array.reshape((1, self.number_of_bands * 9 + 2))
                    df_SS_test = pd.concat([df_SS_test, pd.DataFrame(array, columns=columns_test)], ignore_index=True)

        return df_SS_train, df_SS_test

    def gather_train_and_test(self, df_train, df_test):
        return df_train.iloc[:, :-1], df_train.iloc[:, -1], df_test

    def fill_SS_nan(self, train_model_function, param_grid, df_train, df_test):
        X_train, y_train, X_test = self.gather_train_and_test(df_train, df_test)
        LR_model, results_df = train_model_function(param_grid, X_train, y_train)

        result_filled_image = np.copy(self.cloudy_image)

        for index, row in X_test.iterrows():
            i = int(row['location i'])
            j = int(row['location j'])

            features = row[:-2]
            predicted_value = LR_model.predict(features.values.reshape(1, -1))[0]

            result_filled_image[i, j] = round(predicted_value, 3)

        return result_filled_image, results_df

    def train_RF(self, param_grid, X_train, y_train):
        results_df = pd.DataFrame()

        start = time.time()
        rf_model = RandomForestRegressor(random_state=42)
        print("Fit Start")
        rf_model.fit(X_train, y_train)
        print("Fit End")
        end = time.time()
        print(f'training model time is {round((end - start) / 60, 3)}')

        return rf_model, results_df
