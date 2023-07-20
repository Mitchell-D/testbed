
from pathlib import Path
import numpy as np
import pickle as pkl
from datetime import datetime

from SparseTimeGrid import SparseTimeGrid
from GeoTimeSeries import GeoTimeSeries
#from basic_lstm import lstm_static_bidir

def split_sequence(sequence, n_steps):
    """
    function that splits a dataset sequence into input data and labels
    """
    X = []
    for i in range(sequence.shape[0]):
        if (i + n_steps) >= sequence.shape[0]:
            break
        # Divide sequence between data (input) and labels (output)
        X.append(sequence[i: i + n_steps])
    return np.array(X)

def init_stg(static_pkl:Path):
    """
    Load static datasets from pkl created by nldas_static_netcdf.py into a new
    SparseTimeGrid object
    """
    static = pkl.load(static_pkl.open("rb"))
    stg = SparseTimeGrid(*static["geo"][::-1])
    stg.add_data_dir("data/GTS")
    stg.add_static("sand_pct", static["soil_comp"][:,:,0])
    stg.add_static("silt_pct", static["soil_comp"][:,:,1])
    stg.add_static("clay_pct", static["soil_comp"][:,:,2])
    stg.add_static("veg_type_ints", static["veg_type_ints"])
    stg.add_static("soil_type_ints", static["soil_type_ints"])
    for i in range(len(static["params_info"])):
        stg.add_static(
                static["params_info"][i]["standard_name"].replace(" ","_"),
                static["params"][i])
    return stg

if __name__=="__main__":
    data_dir = Path("data")
    static_pkl = data_dir.joinpath("static/nldas2_static_all.pkl")
    feature_labels = ["APCP", "CAPE", "DLWRF", "DSWRF", "PEVAP",
                      "PRES", "SPFH", "TMP", "SOILM-0-10"]
    '''
    feature_labels = ["APCP", "CAPE", "DLWRF", "DSWRF", "PEVAP",
                      "PRES", "SPFH", "TMP", "SOILM-0-10", "SOILM-10-40",
                      "SOILM-40-100", "SOILM-100-200"]
    '''
    '''
    static_labels = ['porosity', 'field_capacity', 'wilting_point',
                     'b_parameter', 'matric_potential',
                     'hydraulic_conductivity']
    '''
    static_labels = ["sand_pct", "silt_pct", "clay_pct"]
    truth_feature_label = "SOILM-0-10"
    dataset_pkl = data_dir.joinpath("model_ready/lstm-2-allsoil.pkl")
    #window_size = 24 # lstm-3
    window_size = 48 # lstm-2

    """ Initialize a SparseTimeGrid object with static datasets """
    stg = init_stg(static_pkl)

    """
    Get a list of GeoTimeSeries objects corresponding to the features for each
    valid pixel for both the training and validation sets.

    lstm-2:
        - 48 lookback window
        - soil type 4 (sandy loam) only;
        - t20180401-20180901; v20210401-20210901
    lstm-2-allsoil:
        - 48hr lookback window
        - t20180401-20180901; v20210401-20210901
    lstm-3:
        - soil type 4 (sandy loam) only;
        - Add all soilm layers as features
        - t20180401-20180901; v20210401-20210901
    """
    gtss_train = list(stg.search(
            time_range=(datetime(year=2018, month=4, day=1),
                        datetime(year=2018, month=9, day=1)),
            #static={"soil_type_ints":4}, # sandy loam
            group_pixels=True
            ).values())
    gtss_val = list(stg.search(
            time_range=(datetime(year=2021, month=4, day=1),
                        datetime(year=2021, month=9, day=1)),
            #static={"soil_type_ints":4}, # sandy loam
            group_pixels=True
            ).values())

    """
    Get training datasets for truth, feature, and static data shaped so that
        features: (t,w,f) for t times, w window size, f features
        static: (t,s) for t times, and s static values
        truth: (t,) for t times
    """
    t_feats = []
    t_static = []
    t_truth = []
    for i in range(len(gtss_train)):
        truth_array = next(g.data for g in gtss_train[i]
                           if g.flabel == truth_feature_label)
        # Skip arrays with all maskked values.
        if all(truth_array==9999.):
            continue
        px_feats = [next(split_sequence(g.data, window_size)
                         for g in gtss_train[i] if g.flabel == feat)
                    for feat in feature_labels]
        px_feats = np.dstack(px_feats)
        px_static = np.array([stg.static[k][gtss_train[i][0].idx]
                              for k in static_labels]).T
        # Get an array with each static value from the SparseTimeGrid for
        # this pixel, extended to cover the same number of time steps
        px_static = np.vstack([px_static for i in range(px_feats.shape[0])])
        # Extract the truth value corresponding to each time step
        px_truth = truth_array[window_size:window_size+px_feats.shape[0]]
        t_feats.append(px_feats)
        t_static.append(px_static)
        t_truth.append(px_truth)
    t_feats = np.concatenate(t_feats)
    t_static = np.concatenate(t_static)
    t_truth = np.concatenate(t_truth)

    """
    Get validation datasets for truth, feature, and static data shaped so that
        features: (t,w,f) for t times, w window size, f features
        static: (t,s) for t times, and s static values
        truth: (t,) for t times
    """
    v_feats = []
    v_static = []
    v_truth = []
    for i in range(len(gtss_val)):
        truth_array = next(g.data for g in gtss_train[i]
                           if g.flabel == truth_feature_label)
        # Skip arrays with all maskked values.
        if all(truth_array==9999.):
            continue
        # Get an ordered list of 'split' time series for each feature
        px_feats = [next(split_sequence(g.data, window_size)
                         for g in gtss_train[i] if g.flabel == feat)
                    for feat in feature_labels]
        px_feats = np.dstack(px_feats)
        # Get an array with each static value from the SparseTimeGrid for
        # this pixel, extended to cover the same number of time steps
        px_static = np.array([stg.static[k][gtss_train[i][0].idx]
                              for k in static_labels]).T
        px_static = np.vstack([px_static for i in range(px_feats.shape[0])])
        # Extract truth values corresponding to each time step
        px_truth = truth_array[window_size:window_size+px_feats.shape[0]]
        v_feats.append(px_feats)
        v_static.append(px_static)
        v_truth.append(px_truth)
    v_feats = np.concatenate(v_feats)
    v_static = np.concatenate(v_static)
    v_truth = np.concatenate(v_truth)

    """
    Gather a dictionary of mean and standard devia corresponding to each
    feature and static data type, and Gaussian-normalize the data for training.
    """
    norm_scales = {}
    # Noramlize feature data
    for i in range(len(feature_labels)):
        mean = np.average(t_feats[:,0,i])
        stdev = np.std(t_feats[:,0,i])
        t_feats[:,:,i] -= mean
        t_feats[:,:,i] /= stdev
        v_feats[:,:,i] -= mean
        v_feats[:,:,i] /= stdev
        norm_scales[feature_labels[i]] = (mean, stdev)
    # Noramlize static data
    for i in range(len(static_labels)):
        mean = np.average(t_static[:,i])
        stdev = np.std(t_static[:,i])
        t_static[:,i] -= mean
        t_static[:,i] /= stdev
        v_static[:,i] -= mean
        v_static[:,i] /= stdev
        norm_scales[static_labels[i]] = (mean, stdev)
    # Normalize truth data
    t_truth -= norm_scales[truth_feature_label][0]
    t_truth /= norm_scales[truth_feature_label][1]
    v_truth -= norm_scales[truth_feature_label][0]
    v_truth /= norm_scales[truth_feature_label][1]

    print(t_feats.shape, t_static.shape, t_truth.shape)
    print(v_feats.shape, v_static.shape, v_truth.shape)
    print(norm_scales)

    pkl.dump({"training":(t_feats, t_static, t_truth),
              "validation":(v_feats, v_static, v_truth),
              "scales":norm_scales},
             dataset_pkl.open("wb"))
