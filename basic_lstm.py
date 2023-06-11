import pickle as pkl
from pathlib import Path
import numpy as np

# It's temporary thanks to warnings from the conda build of tensorflow I need
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

from aes670hw2 import enhance as enh
from aes670hw2 import guitools as gt
from aes670hw2 import geo_plot as gp

def build_lstm():
    """
    Input tensor should be shaped like:
    (batch size, window size, feature dimensions)

    Output tensor should be shaped like:
    (batch size, output dimensions)
    """
    nldas1D = Sequential()

    nldas1D.add(InputLayer((9, 1)))
    nldas1D.add(LSTM(64))
    nldas1D.add(Dense(8, 'relu'))
    nldas1D.add(Dense(1, 'linear'))

    nldas1D.summary()
    return nldas1D

if __name__=="__main__":
    """
    1D datasets are produced by build_1d_dataset, and are formatted as such:
    {
        "feature":ndarray shaped like (timesteps, pixels, feature_bands),
        "truth":ndarray shaped like (timesteps, pixels, output_bands),
        "static":ndarray shaped like (1, pixels, static_datasets),
        "info":{
            "feature":List of dicts for each input band (size feature_bands),
            "truth":List of dicts for each out band (size output_bands),
            "static":List of dicts for static data (size static_datasets),
        }
        "geo":ndarray shaped like (nlat, nlon) for coordinates
        "pixels":List of 2-tuples corresponding to indeces of each pixel.
    }
    """
    debug = True
    data_dir = Path("data")
    fig_dir = Path("figures")
    # set_label denotes a dataset of unique selected pixels
    set_label = "silty-loam"
    data_pkl = data_dir.joinpath(Path("1D/silty-loam_2019.pkl"))

    data_dict = pkl.load(data_pkl.open("rb"))
    print(data_dict.keys())
    build_lstm()
