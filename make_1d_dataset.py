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
from datetime import datetime
from datetime import timedelta

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
    Thus cycles of length 3*2 + (6+3+3) = 18 will be extracted from the time
    series.

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
    t0 = datetime(year=2019, month=1, day=1, hour=0)
    tf = datetime(year=2020, month=1, day=1, hour=0)
    timesteps = [t0+timedelta(hours=hours) for hours in
                 range(int((tf-t0).total_seconds() // 3600 ))]

    """ Set the output pkl path """
    # output_pkl_path = None
    # The output pkl corresponds to the dictionary of info pertaining to the
    # entire contiguous timeseries.
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
    data_dict_1d = make_dataset_1d(
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

    #'''
    """
    Split the the data into epochs and prepare for training.
    This should be generalized into a module
    """
    timesteps = data_dict_1d["timesteps"]
    # For now, select only spring and summer months
    t0 = datetime(year=2019, month=4, day=1, hour=0)
    dt = timedelta(hours=1)
    num_cycles = 4
    training_size = 24*28
    validation_size = 24*7
    testing_size = 24*7
    window_size = 24*2
    cycle_size = 3*window_size+training_size+validation_size+testing_size
    tf = t0 + dt * cycle_size * num_cycles

    # Copy static datasets across timesteps and append them as new features.
    # Features are still in (timestep, pixel, feature) format, as provided
    # in the 1D dataset dictionary
    features = data_dict_1d["feature"]
    static = np.vstack([data_dict_1d["static"]
                        for i in range(features.shape[0])])
    # Subset all relevant datasets to the time constraint
    sub_slice = slice(timesteps.index(t0), timesteps.index(tf))
    truth = data_dict_1d["truth"][sub_slice]
    features = np.dstack((features, static))[sub_slice]
    timesteps = timesteps[sub_slice]

    # Each cycle covers 1152 time steps
    print("cycle size:", cycle_size)
    print("features/truth shape:", features.shape, truth.shape)
    """
    Note: batch, validation, and testing size refer to the number of
    continuous chronological samples extracted for each type PER CYCLE

    In the following code, 't' represents training, 'v' represents validation,
    and 's' represents testing data

    Uses sliding window method to construct an array like
    (batch_size, window, features) for each cycle in the feature data
    (batch_size, features) for each cycle in the truth data,
    and a list of timesteps for each sample

    This should be heavily consolidated into dataset curation methods later.
    """
    alldata = {"training":  {"feature":[],"truth":[],"time":[]},
               "validation":{"feature":[],"truth":[],"time":[]},
               "testing":   {"feature":[],"truth":[],"time":[]}}
    window_slide = lambda start,pos,wdw: slice(start+pos,start+pos+wdw)
    for i in range(num_cycles):
        # Determine window start index and corresponding time range
        t_start = i*cycle_size
        t_times = [timesteps[window_slide(t_start,j,window_size)][-1]+dt
                   for j in range(training_size)]
        # Use the sliding window method to build feature dataset
        t_feat = np.vstack([
            np.expand_dims(features[window_slide(t_start,j,window_size)],0)
            for j in range(training_size)])
        t_truth = truth[t_start+window_size:
                        t_start+window_size+training_size]
        # Append the pixels dimension of each dataset along the first axis
        t_feat = np.vstack([t_feat[:,:,i] for i in range(t_feat.shape[2])])
        t_truth = np.vstack([t_truth[:,i] for i in range(t_truth.shape[1])])
        print(f"\ntrain: {t_times[0]} - {t_times[-1]}",
              t_feat.shape,t_truth.shape)
        alldata["training"]["feature"].append(t_feat)
        alldata["training"]["truth"].append(t_truth)
        alldata["training"]["time"].append(t_times)

        # Determine window start index and corresponding time range
        v_start = t_start+training_size+window_size
        v_times = [timesteps[window_slide(v_start,j,window_size)][-1]+dt
                   for j in range(validation_size)]
        # Use the sliding window method to build feature dataset
        v_feat = np.vstack([
            np.expand_dims(features[window_slide(v_start,j,window_size)],0)
            for j in range(validation_size)])
        v_truth = truth[v_start+window_size:
                        v_start+window_size+validation_size]
        # Append the pixels dimension of each dataset along the first axis
        v_feat = np.vstack([v_feat[:,:,i] for i in range(v_feat.shape[2])])
        v_truth = np.vstack([v_truth[:,i] for i in range(v_truth.shape[1])])
        print(f"valid: {v_times[0]} - {v_times[-1]}",
              v_feat.shape,v_truth.shape)
        alldata["validation"]["feature"].append(v_feat)
        alldata["validation"]["truth"].append(v_truth)
        alldata["validation"]["time"].append(v_times)

        # Determine window start index and corresponding time range
        s_start = v_start+validation_size+window_size
        s_times = [timesteps[window_slide(s_start,j,window_size)][-1]+dt
                   for j in range(testing_size)]
        # Use the sliding window method to build feature dataset
        s_feat = np.vstack([
            np.expand_dims(features[window_slide(s_start,j,window_size)],0)
            for j in range(testing_size)])
        s_truth = truth[s_start+window_size:
                        s_start+window_size+testing_size]
        # Append the pixels dimension of each dataset along the first axis
        s_feat = np.vstack([s_feat[:,:,i] for i in range(s_feat.shape[2])])
        s_truth = np.vstack([s_truth[:,i] for i in range(s_truth.shape[1])])
        print(f"test:  {s_times[0]} - {s_times[-1]}",
              s_feat.shape, s_truth.shape)
        alldata["testing"]["feature"].append(s_feat)
        alldata["testing"]["truth"].append(s_truth)
        alldata["testing"]["time"].append(s_times)
    #'''

    '''
    # For set1, just take the first cycle
    training_pkl = Path(f"data/model_data/silty-loam_set1_training.pkl")
    pkl.dump(tuple([alldata["training"][k][0]
                    for k in ("feature", "truth", "time")]),
             training_pkl.open("wb"))

    validation_pkl = Path(f"data/model_data/silty-loam_set1_validation.pkl")
    pkl.dump(tuple([alldata["validation"][k][0]
                    for k in ("feature", "truth", "time")]),
             validation_pkl.open("wb"))

    testing_pkl = Path(f"data/model_data/silty-loam_set1_testing.pkl")
    pkl.dump(tuple([alldata["testing"][k][0]
                    for k in ("feature", "truth", "time")]),
             testing_pkl.open("wb"))
    '''

    exit(0)
    #'''
    # For set3, stack all cycles along the timestep axis, so that each batch
    # contains every continuous time series (cycle) for all pixels.
    t_pkl = Path(f"data/model_data/silty-loam_set3_training.pkl")
    v_pkl = Path(f"data/model_data/silty-loam_set3_validation.pkl")
    s_pkl = Path(f"data/model_data/silty-loam_set3_testing.pkl")

    t_combined = tuple([np.vstack(alldata["training"][k])
                        for k in ("feature", "truth", "time")])
    pkl.dump(t_combined, t_pkl.open("wb"))

    v_combined = tuple([np.vstack(alldata["validation"][k])
                        for k in ("feature", "truth", "time")])
    pkl.dump(v_combined, v_pkl.open("wb"))

    s_combined = tuple([np.vstack(alldata["testing"][k])
                        for k in ("feature", "truth", "time")])
    pkl.dump(s_combined, s_pkl.open("wb"))
    #'''
