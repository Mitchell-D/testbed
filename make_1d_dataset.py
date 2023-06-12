"""
Procedural script that aggregates:
 1. NLDAS/Noah-LSM data extracted per pixel/timestep by get_nldas2_1d.py
 2. Soil/vegetation type, and other parameters that don't vary per-pixel, from
    data extracted by nldas_static_netcdf.py.
 3. List of dictionaries containing information for each NLDAS forcing,
    Noah-LSM output, and static data field. The index of the list corresponds
    to the 3rd index of the corresponding dataset.
 3. (nlat,nlon,2) array of geographic coordinates
 4. List of 2-tuple pixel indeces indicating the pixels selected by the user
    in the get_nldas2_1d.py script
"""
import pickle as pkl
from pathlib import Path
import numpy as np
from datetime import datetime as dt
from datetime import timedelta as td

from aes670hw2 import enhance as enh
from aes670hw2 import guitools as gt
from aes670hw2 import geo_plot as gp

def make_dataset_1d(
        feature_data:np.ndarray, truth_data:np.ndarray, timesteps:list,
        static_data:np.ndarray, latitude:np.ndarray, longitude:np.ndarray,
        feature_info:list, truth_info:list, static_info:list, pixels:list,
        pkl_path:Path=None):
    """
    As long as I'm training or testing 1D models that vary with respect to
    time series and static data selected from pixels on a 2d grid, I can
    store everything I need in a dictionary such that:

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
        "timesteps":List of equal-interval datetimes for each timestep.
    }

    This array is intended for data validation and reliable generalization,
    and should be adapted into a product class at some point in the future.

    By my own convention, the list indeces of the info dict for each dataset
    corresponds to the index of the third axis index of the data it describes.

    :@param: See corresponding fields in docstring above
    :@param pkl_path: if valid path, writes the returned dataset to a binary
        pkl at the provided location.
    :@return: 1D dataset dict with fields adhering to the above format
    """
    # feature data must be shaped like truth (label) data
    assert feature_data.shape[:2] == truth_data.shape[:2]
    # static data must have the same number of pixels as label/truth data
    assert static_data.shape[1] == truth_data.shape[1]
    # latitude and longitude grids must be stackable
    assert latitude.shape == longitude.shape
    # static data should have a first dimension of 1 since they are invariant
    # along time steps. This ensures generality with feature/truth data.
    assert static_data.shape[0] == 1
    # pixels must be 2-tuples
    assert all([ len(px)==2 for px in pixels])
    # There must be a timestep datetime for each feature/truth time
    assert len(timesteps)== feature_data.shape[0]
    # Timesteps must increase linearly
    tdelta = timesteps[1]-timesteps[0]
    assert all(map(lambda t:t[1]-t[0]==tdelta,
                   zip(timesteps[:-1],timesteps[1:])))

    # Ensure all pixels are on the coordinate grid
    ypx, xpx = zip(*pixels)
    assert all([y<latitude.shape[0] for y in ypx])
    assert all([x<latitude.shape[1] for x in xpx])
    dataset = {
            "feature":feature_data,
            "truth":truth_data,
            "static":static_data,
            "info":{
                "feature":feature_info,
                "truth":truth_info,
                "static":static_info,
                },
            "geo":np.dstack((latitude, longitude)),
            "pixels":pixels,
            "timesteps":timesteps,
            }
    if pkl_path:
        pkl.dump(dataset,pkl_path.open("wb"))
    return dataset

def cycle_split_dataset_1d(
        timeseries_size:int, training_size:int,validation_size:int,
        testing_size:int, window_size:int):
    """
    Method that generates a list of indeces for training, validation, and
    testing datasets within a continuous equal-interval time series dataset
    such that

    :@param timeseries_size: Integer length in number of timesteps of the
        relevant continuous equal-interval monotonically increasing time series
        data.

    For a dataset with...
     - timeseries size 24
     - (w) window size 2
     - (a) training size 6
     - (b) validation size 3
     - (c) testing size 3

    3*2 timesteps are dedicated to initializing the window of each subset,
    and 6+3+3 timesteps are part of the training, validation, and testing.

    Cycles of length 3*2 + (6+3+3) = 18 will be extracted from the time series.

    [ ww:aaaaaa|ww:bbb|ww:ccc || ww:aaaaaa|ww:bbb|ww:ccc ] xxxxxx
      -------(cycle 1)-------    -------(cycle 2)-------   --(leftover)--
    """
    pass

if __name__=="__main__":
    debug = True
    data_dir = Path("data")
    fig_dir = Path("figures")

    """ """
    # set_label denotes a dataset of unique selected pixels, and is
    # the first underscore-separated field of filenames, by my convention.
    set_label = "silty-loam"
    nldas_pkl = data_dir.joinpath(
            f"1D/{set_label}_nldas2_all-forcings_2019.pkl")
    noahlsm_pkl = data_dir.joinpath(
            f"1D/{set_label}_noahlsm_all-fields_2019.pkl")
    static_pkl = data_dir.joinpath("static/nldas2_static_all.pkl")

    # Load the timesteps as each hour within the provided time range.
    # This information couldn't be easily pickled before.
    t0 = dt(year=2019, month=1, day=1, hour=0)
    tf = dt(year=2020, month=1, day=1, hour=0)
    timesteps = [t0+td(hours=hours) for hours in
                 range(int((tf-t0).total_seconds() // 3600 ))]

    """ Set the output pkl path """
    # output_pkl_path = None
    output_pkl_path = Path(f"data/1D/{set_label}_2019_lsoil.pkl")
    noahlsm_records = (30, 31, 32, 33)
    nldas_records = tuple(range(1,12))

    """ Load static information  """
    static_dict = pkl.load(static_pkl.open("rb"))
    lat, lon = static_dict["geo"]
    soil_comp = static_dict["soil_comp"] # shape: (224,464,3)
    veg_ints = static_dict["veg_type_ints"] # shape: (224, 464)
    params = np.dstack(static_dict["params"]) # shape: (224,464,6)
    params_info = np.dstack(static_dict["params_info"])
    """
    Load the 1D NLDAS forcings and Noah-LSM soil moisture bins, which are
    expected to be formatted as a tuple (array, pixels, info) where 'array' is
    a (t,p,b)-shaped  array for t times, p pixels, and b 'bands' (features),
    pixels is a length 'p' list of 2-tuple indeces like (j,i) corresponding to
    each selecetd pixel, and info is a length 'b' list of dicts with meta-info
    corresponding to each data field, or 'feature'.
    """
    # Load selected Noah-LSM records, referenced with wgrib record numbers
    noahlsm,_,noahlsm_info = pkl.load(noahlsm_pkl.open("rb"))
    noahlsm = np.dstack([noahlsm[:,:,i] for i in range(noahlsm.shape[2])
                         if noahlsm_info[i]["record"] in noahlsm_records])

    # Load selected NLDAS-2 records, referenced with wgrib record numbers
    nldas,pixels,nldas_info = pkl.load(nldas_pkl.open("rb"))
    nldas = np.dstack([nldas[:,:,i] for i in range(nldas.shape[2])
                       if nldas_info[i]["record"] in nldas_records])

    # Parse the pixel values from the 2d static arrays
    static = np.dstack((params, soil_comp, veg_ints))[tuple(zip(*pixels))]
    static = np.expand_dims(static,0)

    """
    Restore the 'curated' dataset pkl by combining Noah-LSM and NLDAS-2 time
    series with the pertainent static datasets, coordinates, and information
    dictionaries (from wgrib, etc).
    """
    print(f"Writing 1D dataset to {output_pkl_path.as_posix()}")
    dataset = make_dataset_1d(
            feature_data=nldas,
            truth_data=noahlsm,
            static_data=static,
            latitude=lat,
            longitude=lon,
            timesteps=timesteps,
            feature_info=nldas_info,
            truth_info=noahlsm_info,
            static_info=params_info,
            pixels=pixels,
            #pkl_path=output_pkl_path,
            )

