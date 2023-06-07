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


def build_lstm():
    nldas1D = Sequential()

    nldas1D.add(InputLayer((9, 1)))
    nldas1D.add(LSTM(64))
    nldas1D.add(Dense(8, 'relu'))
    nldas1D.add(Dense(1, 'linear'))

    nldas1D.summary()
    return nldas1D

def make_dataset_pkl(
        input_data:np.ndarray, truth_data:np.ndarray, static_data:np.ndarray,
        latitude:np.ndarray, longitude:np.ndarray,
        input_info:dict, truth_info:dict, static_info:dict):
    """
    As long as I'm training or testing 1D models that vary with respect to
    time series and static data selected from pixels on a 2d grid, I can
    store everything I need in a dictionary such that

    {
        "input":ndarray shaped like (timesteps, pixels, input_bands),
        "truth":ndarray shaped like (timesteps, pixels, output_bands),
        "static":ndarray shaped like (1, pixels, static_datasets),
        "geo":tuple like (lat, lon) for equal-sized 2d ndarrays for each.
        "info":{
            "input":List of info dicts for each input band (size input_bands),
            "truth":List of info dicts for each out band (size output_bands),
            "static":List of info dicts for static data (size static_datasets),
        }
    }
    """
    assert input_data.shape[:2] == truth_data.shape[:2] == \
            static_data.shape[:2] == latitude.shape[:2] == longitude.shape[:2]
    print(input_data.shape)

if __name__=="__main__":
    debug = True
    data_dir = Path("data")
    fig_dir = Path("figures")
    set_label = "silty-loam"
    nldas_pkl = data_dir.joinpath(f"1D/{set_label}_nldas2_forcings_2019.pkl")
    noahlsm_pkl = data_dir.joinpath(f"1D/{set_label}_noahlsm_soilm_2019.pkl")
    pixels, nldas = pkl.load(nldas_pkl.open("rb"))
    _, noahlsm = pkl.load(noahlsm_pkl.open("rb"))
    soilm = noahlsm[:,:,25:29]
    print(pixels, nldas.shape, noahlsm.shape)

    """
    Restore the 'curated' dataset pkl by combining Noah-LSM and NLDAS-2 time
    series with the pertainent static datasets, coordinates, and information
    dictionaries (from wgrib, etc).
    """
    from records_nldas import records_nldas
    from records_noahlsm import records_noahlsm
    make_dataset_pkl(
            input_data=nldas,
            truth_data=
            )

    build_lstm()
