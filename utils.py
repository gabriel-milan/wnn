from encoders import ThermometerEncoder, CircularThermometerEncoder

from os import path

import cv2 as cv
import numpy as np
import pandas as pd
from skimage.filters import threshold_niblack, threshold_sauvola
from sklearn.preprocessing import normalize


def load_data() -> tuple:
    """Loads shower shape and rings data"""

    def get_df(fname: str, target: int) -> tuple:
        m_data = dict(np.load(path.join("data/", fname)))
        m_df = pd.DataFrame(data=m_data["data"], columns=m_data["features"])
        m_df["target"] = target
        return m_df

    def parse_etbin(et: float) -> int:
        et /= 1000
        if (et < 20):
            return 0
        elif (et < 30):
            return 1
        elif (et < 40):
            return 2
        elif (et < 50):
            return 3
        else:
            return 4

    def split_etbin(df: pd.DataFrame, feature_cols: list) -> dict:
        """Splits dataframe into 5 equal-sized bins and only feature columns"""
        d = {}
        for i in range(5):
            d[i] = df[df["etbin"] == i][feature_cols].values
        return d

    def normalize_df_dict(df_dict: dict) -> dict:
        """Normalizes dataframe values"""
        for i in range(5):
            df_dict[i] = normalize(df_dict[i], axis=1, norm="l1")
        return df_dict

    # Load full data
    zee: pd.DataFrame = get_df('zee.npz', 1)
    jets: pd.DataFrame = get_df('jets.npz', 0)

    # Filter eta region
    zee = zee[(zee['eta'] > -0.8) & (zee['eta'] < 0.8)]
    jets = jets[(jets['eta'] > -0.8) & (jets['eta'] < 0.8)]

    # Merge data, dropping NaNs
    data = zee.append(jets, ignore_index=True)
    data.dropna(inplace=True)

    # Parse E_T bin
    data['etbin'] = data['et'].apply(parse_etbin)

    # Split shower shape data and ring data
    shower_shape_features = ['eratio', 'reta',
                             'rphi', 'rhad', 'f1', 'f3', 'weta2']
    ring_features = ['ring_{}'.format(x) for x in range(100)]

    X_ss: dict = normalize_df_dict(split_etbin(data, shower_shape_features))
    X_ring: dict = normalize_df_dict(split_etbin(data, ring_features))
    y: dict = split_etbin(data, ['target'])

    # Flatten y
    y = {k: [str(val) for val in v.ravel()] for k, v in y.items()}

    return X_ss, X_ring, y


class Binarizer():
    """Utility for KMNIST binarization"""

    def basic_bin(self, arr: np.ndarray, threshold: int = 128) -> list:
        return [list(np.where(x < threshold, 0, 1).flatten()) for x in arr]

    def simple_thermometer(self, arr: np.ndarray, minimum: int = 0, maximum: int = 255, resolution: int = 25) -> list:
        therm = ThermometerEncoder(
            maximum=maximum, minimum=minimum, resolution=resolution)
        return [np.uint8(therm.encode(x)).flatten() for x in arr]

    def circular_thermometer(self, arr: np.ndarray, minimum: int = 0, maximum: int = 255, resolution: int = 20) -> list:
        therm = CircularThermometerEncoder(
            maximum=maximum, minimum=minimum, resolution=resolution)
        return [np.uint8(therm.encode(x)).flatten() for x in arr]

    def sauvola(self, arr: np.ndarray, window_size: int = 11) -> list:
        bin_imgs = list()
        for x in arr:
            thresh_s = threshold_sauvola(x, window_size=window_size)
            binary_s = np.array(x > thresh_s, dtype=int)
            bin_imgs.append(binary_s.flatten())
        return bin_imgs

    def niblack(self, arr: np.ndarray, window_size: int = 11, k: float = 0.8) -> list:
        bin_imgs = list()
        for x in arr:
            thresh_n = threshold_niblack(x, window_size=window_size, k=k)
            binary_n = np.array(x > thresh_n, dtype=int)
            bin_imgs.append(binary_n.flatten())
        return bin_imgs

    def adaptive_thresh_mean(self, arr: np.ndarray, window_size: int = 11, constant_c: int = 2) -> list:
        return [cv.adaptiveThreshold(x, 1, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, window_size, constant_c).flatten() for x in arr]

    def adaptive_thresh_gaussian(self, arr: np.ndarray, window_size: int = 11, constant_c: int = 2) -> list:
        return [cv.adaptiveThreshold(x, 1, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, window_size, constant_c).flatten() for x in arr]
