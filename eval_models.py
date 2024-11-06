import numpy as np
import pickle as pkl
import random as rand
import json
import h5py
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
import matplotlib.pyplot as plt
import tensorflow as tf

import model_methods as mm
import tracktrain as tt
import generators

def add_norm_layers(md:tt.ModelDir, weights_file:str=None,
        dynamic_norm_coeffs:dict={}, static_norm_coeffs:dict={},
        save_new_model=False):
    """
    Wrap a model with layers linearly scaling the window, horizon, and static
    inputs before and predicted outputs after running the model such that

    x' = (x - b) / m    and    y' = y * m + b

    where (m,b) are parameters specified per-feature in the coefficient dicts

    :@param ModelDir: ModelDir object for a trained model.
    :@param weights_file: Optional weights file to load. if left to None,
        the '_final.weights.h5' model is used.
    :@param dynamic_norm_coeffs: Dictionary mapping feature names to
        normalization coefficients for time-varying data
    :@param static_norm_coeffs: Dictionary mapping feature names to
        normalization coefficients for static data.
    :@param save_new_model: If True, the new model weights including
        normalization are saved to a new file with name ending "_normed"
    """
    if not weights_file is None:
        weights_file = Path(weights_file).name

    model = md.load_weights(weights_path=weights_file)
    w_norm = np.array([
        tuple(dynamic_norm_coeffs[k])
        if k in dynamic_norm_coeffs.keys() else (0,1)
        for k in md.config["feats"]["window_feats"]
        ])[np.newaxis,np.newaxis,:]
    h_norm = np.array([
        tuple(dynamic_norm_coeffs[k])
        if k in dynamic_norm_coeffs.keys() else (0,1)
        for k in md.config["feats"]["horizon_feats"]
        ])[np.newaxis,np.newaxis,:]
    s_norm = np.array([
        tuple(static_norm_coeffs[k])
        if k in static_norm_coeffs.keys() else (0,1)
        for k in md.config["feats"]["static_feats"]
        ])[np.newaxis,:]
    p_norm = np.array([
        tuple(dynamic_norm_coeffs[k])
        if k in dynamic_norm_coeffs.keys() else (0,1)
        for k in md.config["feats"]["pred_feats"]
        ])[np.newaxis,:]

    w_in = tf.keras.Input(
            shape=(
                md.config["model"]["window_size"],
                md.config["model"]["num_window_feats"],),
            name="in_window")
    h_in = tf.keras.Input(
            shape=(
                md.config["model"]["horizon_size"],
                md.config["model"]["num_horizon_feats"],),
            name="in_horizon")
    s_in = tf.keras.Input(
            shape=(md.config["model"]["num_static_feats"],),
            name="in_static")
    si_in = tf.keras.Input(
            shape=(md.config["model"]["num_static_int_feats"],),
            name="in_static_int")

    out = model((
        (w_in-w_norm[...,0])/w_norm[...,1],
        (h_in-h_norm[...,0])/h_norm[...,1],
        (s_in-s_norm[...,0])/s_norm[...,1],
        si_in
        )) * p_norm[...,1] + p_norm[...,0]

    new_model = tf.keras.Model(inputs=(w_in,h_in,s_in,si_in), outputs=out)
    return new_model

def sequence_preds_to_hdf5(model_dir:tt.ModelDir, sequence_generator_args:dict,
        pred_h5_path:Path, weights_file_name:str=None, chunk_size=256,
        gen_batch_size=256, max_batches=None, pred_norm_coeffs:dict={}):
    """
    Evaluates the provided model on a series of sequence files, and stores the
    predictions in a new hdf5 file with a float32 dataset shaped (N,S,F_p) for
    N samples of length S sequences having F_p predicted features, and a uint
    dataset of epoch times shaped (N,S).

    :@param model_dir: tracktrain.ModelDir object for the desired model.
    :@param sequence_generator_args: Dict of arguments sufficient to initialize
        a sequence dataset generator with generators.sequence_dataset().
        The full dict should be json-serializable (ex no Path objects) because
        it will be stored alongside the prediction data as an attribute in
        order to init similar generators in the future.
    :@param pred_h5_path: path to a new hdf5 file storing predictions. Should
        conform to format "pred_{region}_{season}_{timerange}_{model}.h5"
    :@param weights_file_name: Name of the model weights file from the ModelDir
        directory to be used for inference.
    :@param chunk_size: Number of samples per chunk in the new hdf5
    :@param gen_batch_size: Number of samples to draw from the gen at once.
    :@param max_batches: Maximum number of gen batches to store in the file.
    :@param pred_norm_coeffs: Dictionary mapping feature names to 2-tuples
        (mean,stdev) representing linear norm coefficients for dynamic feats.
    """
    ## Generator loop expects times since they will be recorded in the file.
    assert sequence_generator_args.get("yield_times") == True

    if not weights_file_name is None:
        weights_file_name = Path(weights_file_name).name

    model = md.load_weights(weights_path=weights_file_name)

    ## ignore any conditions restricting training
    sequence_generator_args["static_conditions"] = []
    gen = generators.sequence_dataset(**sequence_generator_args)

    p_norm = np.array([
        tuple(pred_norm_coeffs[k])
        if k in pred_norm_coeffs.keys() else (0,1)
        for k in md.config["feats"]["pred_feats"]
        ])[np.newaxis,:]

    ## Create a new h5 file with datasets for the model (residual) predictions,
    ## timesteps, and initial states (last observed predicted states).
    F = h5py.File(
            name=pred_h5_path,
            mode="w-",
            rdcc_nbytes=128*1024**2, ## use a 128MB cache
            )
    output_shape = tuple(gen.element_spec[1].shape)
    P = F.create_dataset(
            name="/data/preds",
            shape=(0, *output_shape),
            maxshape=(None, *output_shape),
            chunks=(chunk_size, *output_shape),
            compression="gzip",
            )
    T = F.create_dataset(
            name="/data/time",
            shape=(0, output_shape[0]),
            maxshape=(None, output_shape[0]),
            chunks=(chunk_size, output_shape[0]),
            compression="gzip"
            )
    Y0 = F.create_dataset(
            name="/data/init_states",
            shape=(0, output_shape[-1]),
            maxshape=(None, output_shape[0]),
            chunks=(chunk_size, output_shape[0]),
            compression="gzip",
            )
    ## Store the generator arguments so the same kind can be re-initialized
    F["data"].attrs["gen_args"] = json.dumps(sequence_generator_args)

    h5idx = 0
    batch_counter = 0
    max_batches = (max_batches, -1)[max_batches is None]
    for (w,h,s,si,t),ys in gen.batch(gen_batch_size):
        ## Normalize the predictions (assumes add_norm_layers not used!!!)
        p = model((w,h,s,si)) * p_norm[...,1]
        ## retain the initial observed state so the residual can be accumulated
        y0 = ys[:,0,:][:,np.newaxis,:] * p_norm[...,1] + p_norm[...,0]
        th = t[:,-p.shape[1]:]

        sample_slice = slice(h5idx, h5idx+ys.shape[0])
        h5idx += ys.shape[0]
        P.resize((h5idx, *p.shape[1:]))
        T.resize((h5idx, th.shape[-1]))
        Y0.resize((h5idx, y0.shape[-1]))

        P[sample_slice,...] = p.numpy()
        T[sample_slice,...] = th.numpy()
        Y0[sample_slice,...] = np.reshape(
                y0.numpy(),(y0.shape[0],y0.shape[-1]))
        #F.flush()

        batch_counter += 1
        if  batch_counter == max_batches:
            break
    F.close()
    return pred_h5_path

def eval_error_horizons(sequence_h5, prediction_h5,
        batch_size=1024, buf_size_mb=128):
    """
    Calculate the state and residual error and approximate variance with
    respect to each forecast horizon in a sequence/prediction hdf5 pair
    """
    param_dict = generators.parse_sequence_params(sequence_h5)
    pred_dict = generators.parse_prediction_params(prediction_h5)
    print(f"horizons {prediction_h5.name}")
    coarseness = pred_dict.get("pred_coarseness", 1)
    gen = generators.gen_sequence_prediction_combos(
            seq_h5=sequence_h5,
            pred_h5=prediction_h5,
            batch_size=batch_size,
            buf_size_mb=buf_size_mb,
            pred_coarseness=coarseness
            )
    counts = None
    es_sum = None
    er_sum = None
    es_var_sum = None
    er_var_sum = None
    for (_, (ys, pr)) in gen:
        ## the predicted state time series
        ps = ys[:,0,:][:,np.newaxis,:] + np.cumsum(pr, axis=1)
        ## Calculate the label residual from labels
        yr = ys[:,1:]-ys[:,:-1]
        ## Calculate the error in the residual and state predictions
        es = ps - ys[:,1:,:]
        er = pr - yr

        es_abs = np.abs(es)
        er_abs = np.abs(er)

        ## Calculate the average absolute error and approximate variance
        ## (approximate since early samples are computed w/ partial avg)
        if counts is None:
            counts = es_abs.shape[0]
            es_sum = np.sum(es_abs, axis=0)
            er_sum = np.sum(er_abs, axis=0)
            es_var_sum = np.sum((es_abs - es_sum/counts)**2, axis=0)
            er_var_sum = np.sum((er_abs - er_sum/counts)**2, axis=0)
        else:
            counts += es_abs.shape[0]
            es_sum += np.sum(es_abs, axis=0)
            er_sum += np.sum(er_abs, axis=0)
            es_var_sum += np.sum((es_abs - es_sum/counts)**2, axis=0)
            er_var_sum += np.sum((er_abs - er_sum/counts)**2, axis=0)
    return {
            "state_avg":es_sum/counts,
            "state_var":es_var_sum/counts,
            "residual_avg":er_sum/counts,
            "residual_var":er_var_sum/counts,
            "counts":counts,
            "feats":pred_dict["pred_feats"],
            "pred_coarseness":coarseness,
            }

def mp_eval_error_horizons(args:tuple):
    return eval_error_horizons(*args)

def eval_joint_hists(
        sequence_h5:Path, prediction_h5:Path,
        pred_state_bounds:dict, pred_residual_bounds:dict, num_bins:int,
        batch_size=1024, buf_size_mb=128, horizon_limit=None,
        get_state_hist=True, get_residual_hist=True):
    """
    Returns (NY,NP,F_p) shaped integer grids of sample magnitude counts binned
    according to each feature's bounds, constructed like a validation grid or a
    joint histogram of label and predicted values such that NY == NP.

    returns a dict formatted like:
    {
        "state_hist":3D array of state counts with label values on the 1st axis
        "state_bounds"2-tuple (mins,maxs) of 1d arrays for state feats
        "residual_hist":3D array of residual counts with labels on the 1st axis
        "residual_bounds":2-tuple (mins,maxs) of 1d arrays for residual feats
    }

    :@param sequence_h5: Path to a sequence hdf5 generated by
        generators.sequence_dataset, having an order that matches the
        provided prediction hdf5
    :@param prediction_h5: Path to a prediction hdf5 from
        sequence_preds_to_hdf5 ordered identically to the provided seq hdf5
    :@param pred_state_bounds: Dictionary mapping prediction feature names to
        inclusive 2-tuple [min,max] bounds for the histogram magnitude extent.
    :@param pred_residual_bounds: Dictionary mapping prediction feature names
        to 2-tuple [min,max] bounds for the histogram magnitude extent.
    :@param num_bins: Number of bins in the histogram between min and max
    :@param batch_size: Number of samples to simultaneously evaluate
    :@param buf_size_mb: hdf5 chunk buffers size. Probably not important here.
    :@param horizon_limit: Maximum number of horizon steps to include in the
        histogram analysis; useful for excluding diverging state values
    :@param get_state_hist: When True, value counts are collected for state
        magnitudes, and included in the returned dictionary.
    :@param get_residual_hist: When True, value counts are collected for
        residual magnitudes, and included in the returned dictionary.
    """
    param_dict = generators.parse_sequence_params(sequence_h5)
    pred_dict = generators.parse_prediction_params(prediction_h5)
    coarseness = pred_dict.get("pred_coarseness", 1)
    print(f"hists {prediction_h5.name}")
    smins,smaxs = zip(*[
        pred_state_bounds[k]
        for k in pred_dict["pred_feats"]
        ])
    rmins,rmaxs = zip(*[
        pred_residual_bounds[k]
        for k in pred_dict["pred_feats"]
        ])
    rmins,rmaxs,smins,smaxs = map(np.array, (rmins, rmaxs, smins, smaxs))
    if horizon_limit is None:
        horizon_limit = param_dict["horizon_size"]

    def _norm_to_idxs(A:np.array, mins, maxs):
        A = (np.clip(A, mins, maxs) - mins) / (maxs - mins)
        A = np.clip(np.floor(A * num_bins), 0, num_bins-1).astype(np.uint32)
        return A

    gen = generators.gen_sequence_prediction_combos(
            seq_h5=sequence_h5,
            pred_h5=prediction_h5,
            batch_size=batch_size,
            buf_size_mb=buf_size_mb,
            pred_coarseness=coarseness,
            )
    r_counts = np.zeros((num_bins,num_bins,rmins.size), dtype=np.uint32)
    s_counts = np.zeros((num_bins,num_bins,smins.size), dtype=np.uint32)
    for (_, (ys, pr)) in gen:
        if get_state_hist:
            ## accumulate the predicted state time series
            ps = ys[:,0,:][:,np.newaxis,:] + np.cumsum(pr, axis=1)

            ys_idxs = np.reshape(
                    _norm_to_idxs(ys[:,1:1+horizon_limit], smins, smaxs),
                    (-1, smins.size))
            ps_idxs = np.reshape(
                    _norm_to_idxs(ps[:,:horizon_limit], smins, smaxs),
                    (-1, smins.size))
            ## Loop since fancy indexing doesn't accumulate repetitions
            for i in range(ys_idxs.shape[0]):
                for j in range(ys_idxs.shape[-1]):
                    s_counts[ys_idxs[i,j],ps_idxs[i,j],j] += 1

        if get_residual_hist:
            ## Calculate the label residual from labels
            yr = ys[:,1:]-ys[:,:-1]
            yr_idxs = np.reshape(
                    _norm_to_idxs(yr[:,1:1+horizon_limit], rmins, rmaxs),
                    (-1, rmins.size))
            pr_idxs = np.reshape(
                    _norm_to_idxs(pr[:,:horizon_limit], rmins, rmaxs),
                    (-1, rmins.size))
            ## Loop since fancy indexing doesn't accumulate repetitions
            for i in range(yr_idxs.shape[0]):
                for j in range(yr_idxs.shape[-1]):
                    r_counts[yr_idxs[i,j],pr_idxs[i,j],j] += 1
    result = {}
    if get_state_hist:
        result["state_hist"] = s_counts
        result["state_bounds"] = (smins, smaxs)
    if get_residual_hist:
        result["residual_hist"] = r_counts
        result["residual_bounds"] = (rmins, rmaxs)
    result["feats"] = pred_dict["pred_feats"]
    return result

def mp_eval_joint_hists(kwargs:dict):
    return eval_joint_hists(**kwargs)

def eval_temporal_error(sequence_h5, prediction_h5,
        batch_size=1024, buf_size_mb=128, horizon_limit=None,
        absolute_error=True):
    """
    Calculate the state and residual error rates of each predicted feature
    with respect to the hourly time of day and the day of year given matching
    sequence input and prediction output hdf5s.

    :@param sequence_h5: Path to a sequence hdf5 generated by
        generators.sequence_dataset, having an order that matches the
        provided prediction hdf5
    :@param prediction_h5: Path to a prediction hdf5 from
        sequence_preds_to_hdf5 ordered identically to the provided seq hdf5
    :@param batch_size: Number of samples to simultaneously evaluate
    :@param buf_size_mb: hdf5 chunk buffers size. Probably not important here.
    :@param horizon_limit: Maximum number of horizon steps to include in the
        histogram analysis; useful for excluding diverging state values
    :@param absolute_error: When True, error magnitudes are absolute, so they
        don't cancel each other out when over and underestimated. When False,
        the errors' actual magnitude (and thus the bias) are returned.
    """
    pred_dict = generators.parse_prediction_params(prediction_h5)
    coarseness = pred_dict.get("pred_coarseness", 1)
    print(f"temporal {prediction_h5.name}")
    gen = generators.gen_sequence_prediction_combos(
            seq_h5=sequence_h5,
            pred_h5=prediction_h5,
            batch_size=batch_size,
            buf_size_mb=buf_size_mb,
            gen_times=True,
            pred_coarseness=coarseness,
            )
    param_dict = generators.parse_sequence_params(sequence_h5)
    if horizon_limit is None:
        horizon_limit = param_dict["horizon_size"]

    doy_r = np.zeros((366, len(pred_dict["pred_feats"])))
    doy_s = np.zeros((366, len(pred_dict["pred_feats"])))
    doy_counts = np.zeros((366, len(pred_dict["pred_feats"])), dtype=np.uint)
    tod_r = np.zeros((24, len(pred_dict["pred_feats"])))
    tod_s = np.zeros((24, len(pred_dict["pred_feats"])))
    tod_counts = np.zeros((24, len(pred_dict["pred_feats"])), dtype=np.uint)
    for ((_,_,_,_,(yt,pt)), (ys, pr)) in gen:
        ps = ys[:,0,:][:,np.newaxis,:] + np.cumsum(pr, axis=1)
        yr = ys[:,1:]-ys[:,:-1]

        times = list(map(
            datetime.fromtimestamp,
            pt.astype(np.uint)[:,:horizon_limit].reshape((-1,))
            ))
        ## Times are reported exactly on the hour, but float rounding can cause
        ## some to be above or below. Add a conditional to account for this.
        tmp_tods = np.array([
            (t.hour+1 if t.minute >= 30 else t.hour)%24 for t in times
            ])
        tmp_doys = np.array([t.timetuple().tm_yday-1 for t in times])

        es = ps - ys[:,1:]
        er = pr - yr
        if absolute_error:
            es,er = map(np.abs,(es,er))
        es = es[:,:horizon_limit].reshape((-1, es.shape[-1]))
        er = er[:,:horizon_limit].reshape((-1, er.shape[-1]))

        for i in range(len(times)):
            doy_s[tmp_doys[i]] += es[i]
            doy_r[tmp_doys[i]] += er[i]
            doy_counts[tmp_doys[i]] += 1
            tod_s[tmp_tods[i]] += es[i]
            tod_r[tmp_tods[i]] += er[i]
            tod_counts[tmp_tods[i]] += 1
    return {
            "doy_state":doy_s,
            "doy_residual":doy_r,
            "doy_counts":doy_counts,
            "tod_state":tod_s,
            "tod_residual":tod_r,
            "tod_counts":tod_counts,
            "feats":pred_dict["pred_feats"],
            }

def mp_eval_temporal_error(kwargs:dict):
    return eval_temporal_error(**kwargs)

def eval_static_error(sequence_h5, prediction_h5,
        batch_size=1024, buf_size_mb=128):
    """
    Calculate the state and residual error rates of each predicted feature
    with respect to the hourly time of day and the day of year given matching
    sequence input and prediction output hdf5s.

    :@param sequence_h5: Path to a sequence hdf5 generated by
        generators.sequence_dataset, having an order that matches the
        provided prediction hdf5
    :@param prediction_h5: Path to a prediction hdf5 from
        sequence_preds_to_hdf5 ordered identically to the provided seq hdf5
    :@param batch_size: Number of samples to simultaneously evaluate
    :@param buf_size_mb: hdf5 chunk buffers size. Probably not important here.
    """
    print(f"static {prediction_h5.name}")
    pred_dict = generators.parse_prediction_params(prediction_h5)
    param_dict = generators.parse_sequence_params(sequence_h5)
    coarseness = pred_dict.get("pred_coarseness", 1)
    gen = generators.gen_sequence_prediction_combos(
            seq_h5=sequence_h5,
            pred_h5=prediction_h5,
            batch_size=batch_size,
            buf_size_mb=buf_size_mb,
            gen_static=True,
            gen_static_int=True,
            pred_coarseness=coarseness,
            )
    ## Gather the indeces of generated soil texture static features
    soil_idxs = tuple(
            param_dict["static_feats"].index(l)
            for l in ("pct_sand", "pct_silt", "pct_clay"))

    ## Soil components to index mapping. Scuffed and slow, I know, but
    ## unfortunately I didn't store integer types alongside sequence samples,
    ## and it's too late to turn back now :(
    soil_mapping = list(map(
        lambda a:np.array(a, dtype=np.float32),
        [
            [0.,   0.,   0.  ],
            [0.92, 0.05, 0.03],
            [0.82, 0.12, 0.06],
            [0.58, 0.32, 0.1 ],
            [0.17, 0.7 , 0.13],
            [0.1 , 0.85, 0.05],
            [0.43, 0.39, 0.18],
            [0.58, 0.15, 0.27],
            [0.1 , 0.56, 0.34],
            [0.32, 0.34, 0.34],
            [0.52, 0.06, 0.42],
            [0.06, 0.47, 0.47],
            [0.22, 0.2 , 0.58],
            ]
        ))

    ## count and error sum matrices shaped for (vegetation, soil)
    counts = np.zeros((14,13))
    err_res = np.zeros((14,13,len(pred_dict["pred_feats"])))
    err_state = np.zeros((14,13,len(pred_dict["pred_feats"])))
    for ((_,_,s,si,_), (ys, pr)) in gen:
        ## the predicted state time series
        ps = ys[:,0,:][:,np.newaxis,:] + np.cumsum(pr, axis=1)
        ## Calculate the label residual from labels
        yr = ys[:,1:]-ys[:,:-1]
        ## Calculate the error in the residual and state predictions
        es = ps - ys[:,1:,:]
        er = pr - yr

        ## Average the error over the full horizon
        es_abs = np.average(np.abs(es), axis=1)
        er_abs = np.average(np.abs(er), axis=1)

        soil_texture = s[...,soil_idxs]
        for i,soil_array in enumerate(soil_mapping):
            ## Get a boolean mask
            m_this_soil = (soil_texture == soil_array).all(axis=1)
            if not np.any(m_this_soil):
                continue
            es_abs_subset = es_abs[m_this_soil]
            er_abs_subset = er_abs[m_this_soil]
            si_subset = si[m_this_soil]
            ## Convert the one-hot encoded vegetation vectors to indeces
            si_idxs = np.argwhere(si_subset)[:,1]
            for j in range(si_idxs.shape[0]):
                err_res[si_idxs[j],i] += er_abs_subset[j]
                err_state[si_idxs[j],i] += es_abs_subset[j]
                counts[si_idxs[j],i] += 1
    return {
            "err_state":err_state,
            "err_residual":err_res,
            "counts":counts,
            "feats":pred_dict["pred_feats"],
            }
def mp_eval_static_error(args:tuple):
    return eval_static_error(*args)

if __name__=="__main__":
    from list_feats import dynamic_coeffs,static_coeffs,derived_feats
    sequence_h5_dir = Path("data/sequences/")
    model_parent_dir = Path("data/models/new")
    pred_h5_dir = Path("data/predictions")
    error_horizons_pkl = Path(f"data/performance/error_horizons.pkl")
    temporal_pkl = Path(f"data/performance/temporal_absolute.pkl")
    hists_pkl = Path(f"data/performance/validation_hists_7d.pkl")
    static_error_pkl = Path(f"data/performance/static_error.pkl")

    ## Evaluate a single model over a series of sequence files, storing the
    ## results in new hdf5 files of predictions in the same order as sequences
    '''
    #model_name = "snow-6"
    model_name = "lstm-rsm-9"
    #weights_file = "lstm-7_095_0.283.weights.h5"
    #weights_file = "lstm-8_091_0.210.weights.h5"
    #weights_file = "lstm-14_099_0.028.weights.h5"
    #weights_file = "lstm-15_101_0.038.weights.h5"
    #weights_file = "lstm-16_505_0.047.weights.h5"
    #weights_file = "lstm-17_235_0.286.weights.h5"
    #weights_file = "lstm-19_191_0.158.weights.h5"
    #weights_file = "lstm-20_353_0.053.weights.h5"
    #weights_file = "lstm-21_522_0.309.weights.h5"
    #weights_file = "lstm-22_339_2.357.weights.h5"
    #weights_file = "lstm-23_217_0.569.weights.h5"
    #weights_file = "lstm-24_401_4.130.weights.h5"
    #weights_file = "lstm-25_624_3.189.weights.h5"
    #weights_file = "lstm-27_577_4.379.weights.h5"
    #weights_file = "snow-4_005_0.532.weights.h5"
    #weights_file = "snow-6_230_0.064.weights.h5"
    #weights_file = "snow-7_069_0.676.weights.h5"
    #weights_file = "lstm-rsm-1_458_0.001.weights.h5"
    #weights_file = "lstm-rsm-6_083_0.013.weights.h5"
    weights_file = "lstm-rsm-9_231_0.003.weights.h5"
    #weights_file = None
    model_label = f"{model_name}-231"

    ## Sequence hdf5s to avoid processing
    seq_h5_ignore = []

    md = tt.ModelDir(
            model_parent_dir.joinpath(model_name),
            custom_model_builders={
                "lstm-s2s":lambda args:mm.get_lstm_s2s(**args),
                })
    ## Get a list of sequence hdf5s which will be independently evaluated
    seq_h5s = mm.get_seq_paths(
            sequence_h5_dir=sequence_h5_dir,
            region_strs=("ne", "nc", "nw", "se", "sc", "sw"),
            #region_strs=("nc",),
            season_strs=("warm", "cold"),
            #season_strs=("cold",),
            #time_strs=("2013-2018"),
            #time_strs=("2018-2023"),
            time_strs=("2018-2021", "2021-2024"),
            )

    ## Ignore min,max values prepended to dynamic coefficients in list_feats
    dynamic_norm_coeffs={k:v[2:] for k,v in dynamic_coeffs}
    ## Arguments sufficient to initialize a generators.sequence_dataset
    seq_gen_args = {
            #"sequence_hdf5s":[p.as_posix() for p in seq_h5s],
            **md.config["feats"],
            "seed":200007221750,
            "frequency":1,
            "sample_on_frequency":True,
            "num_procs":6,
            "block_size":8,
            "buf_size_mb":128.,
            "deterministic":True,
            "shuffle":False,
            "yield_times":True,
            "dynamic_norm_coeffs":dynamic_norm_coeffs,
            "static_norm_coeffs":dict(static_coeffs),
            "derived_feats":derived_feats,
            }
    for h5_path in seq_h5s:
        if Path(h5_path).name in seq_h5_ignore:
            continue
        seq_gen_args["sequence_hdf5s"] = [h5_path]
        _,region,season,time_range = Path(h5_path).stem.split("_")
        pred_h5_path = pred_h5_dir.joinpath(
                f"pred_{region}_{season}_{time_range}_{model_label}.h5")
        sequence_preds_to_hdf5(
                model_dir=md,
                sequence_generator_args=seq_gen_args,
                pred_h5_path=pred_h5_path,
                chunk_size=256,
                gen_batch_size=1024,
                weights_file_name=weights_file,
                pred_norm_coeffs=dynamic_norm_coeffs,
                )
    exit(0)
    '''

    ## Establish sequence and prediction file pairings based on their
    ## underscore-separated naming scheme, which is expected to adhere to:
    ## (sequences file):   {file_type}_{region}_{season}_{period}.h5
    ## (prediction file):  {file_type}_{region}_{season}_{period}_{model}.h5
    #eval_regions = ("sw", "sc", "se")
    eval_regions = ("ne", "nc", "nw", "se", "sc", "sw")
    eval_seasons = ("warm", "cold")
    #eval_periods = ("2018-2023",)
    eval_periods = ("2018-2021", "2021-2024")
    #eval_models = ("lstm-17-235",)
    #eval_models = ("lstm-16-505",)
    #eval_models = ("lstm-19-191", "lstm-20-353")
    #eval_models = ("lstm-21-522", "lstm-22-339")
    #eval_models = ("lstm-23-217",)
    #eval_models = ("lstm-24-401", "lstm-25-624")
    #eval_models = ("snow-4-005",)
    #eval_models = ("snow-7-069",)
    #eval_models = ("lstm-rsm-6-083",)
    eval_models = ("lstm-rsm-9-231",)
    batch_size=2048
    buf_size_mb=128
    num_procs = 7

    """ Match sequence and prediction files, and parse name fields of both """
    seq_pred_files = [
            (s,p,tuple(pt[1:]))
            for s,st in map(
                lambda f:(f,f.stem.split("_")),
                sequence_h5_dir.iterdir())
            for p,pt in map(
                lambda f:(f,f.stem.split("_")),
                pred_h5_dir.iterdir())
            if st[0] == "sequences"
            and pt[0] == "pred"
            and pt[-1] in eval_models
            and st[1:4] == pt[1:4]
            and st[1] in eval_regions
            and st[2] in eval_seasons
            and st[3] in eval_periods
            ]

    ## Generate joint residual and state error histograms
    '''
    residual_bounds = {
            k[4:]:v[:2]
            for k,v in dynamic_coeffs
            if k[:4] == "res_"}
    state_bounds = {k:v[:2] for k,v in dynamic_coeffs}
    kwargs,id_tuples = zip(*[
        ({
            "sequence_h5":s,
            "prediction_h5":p,
            "pred_state_bounds":state_bounds,
            "pred_residual_bounds":residual_bounds,
            "num_bins":128,
            "batch_size":batch_size,
            "buf_size_mb":buf_size_mb,
            "horizon_limit":24*7,
            }, t)
        for s,p,t in seq_pred_files
        ])
    with Pool(num_procs) as pool:
        for i,subdict in enumerate(pool.imap(mp_eval_joint_hists,kwargs)):
            ## Update the histograms pkl with the new model/file results,
            ## distinguished by their id_tuple (region,season,time_range,model)
            if hists_pkl.exists():
                hists = pkl.load(hists_pkl.open("rb"))
            else:
                hists = {}
            hists[id_tuples[i]] = subdict
            pkl.dump(hists, hists_pkl.open("wb"))
    '''

    ## Evaluate the absolute error wrt static parameters for each pair
    '''
    args,id_tuples = zip(*[
            ((sfile, pfile, batch_size, buf_size_mb),id_tuple)
            for sfile, pfile, id_tuple in seq_pred_files
            ])
    with Pool(num_procs) as pool:
        for i,subdict in enumerate(pool.imap(mp_eval_static_error,args)):
            ## Update the error horizons pkl with the new model/file results,
            ## distinguished by their id_tuple (region,season,time_range,model)
            if static_error_pkl.exists():
                static_error = pkl.load(static_error_pkl.open("rb"))
            else:
                static_error = {}
            static_error[id_tuples[i]] = subdict
            pkl.dump(static_error, static_error_pkl.open("wb"))
    '''

    ## Evaluate the absolute error wrt horizon distance for each file pair
    '''
    args,id_tuples = zip(*[
            ((sfile, pfile, batch_size, buf_size_mb),id_tuple)
            for sfile, pfile, id_tuple in seq_pred_files
            ])
    with Pool(num_procs) as pool:
        for i,subdict in enumerate(pool.imap(mp_eval_error_horizons,args)):
            ## Update the error horizons pkl with the new model/file results,
            ## distinguished by their id_tuple (region,season,time_range,model)
            if error_horizons_pkl.exists():
                error_horizons = pkl.load(error_horizons_pkl.open("rb"))
            else:
                error_horizons = {}
            error_horizons[id_tuples[i]] = subdict
            pkl.dump(error_horizons, error_horizons_pkl.open("wb"))
    '''

    ## Calculate error rates with respect to day of year and time of day
    '''
    kwargs,id_tuples = zip(*[
            ({
                "sequence_h5":s,
                "prediction_h5":p,
                "batch_size":batch_size,
                "buf_size_mb":buf_size_mb,
                "horizon_limit":24*7,
                "absolute_error":True,
                }, t)
            for s,p,t in seq_pred_files
            ])
    with Pool(num_procs) as pool:
        for i,subdict in enumerate(pool.imap(mp_eval_temporal_error,kwargs)):
            ## Update the temporal pkl with the new model/file results,
            ## distinguished by their id_tuple (region,season,time_range,model)
            if temporal_pkl.exists():
                temporal = pkl.load(temporal_pkl.open("rb"))
            else:
                temporal = {}
            temporal[id_tuples[i]] = subdict
            pkl.dump(temporal, temporal_pkl.open("wb"))
    '''

    ## combine regions together for bulk statistics
    #'''
    combine_years = ("2018-2021", "2021-2024")
    combine_model = "lstm-rsm-9-231"
    new_key = ("all", "all", "2018-2024", "lstm-rsm-9-231")
    combine_pkl = Path(
            "data/performance/performance-bulk_2018-2024_lstm-rsm-9-231.pkl")

    ## combine histograms
    hists = pkl.load(hists_pkl.open("rb"))
    combine_keys = [k for k in hists.keys()
            if k[3]==combine_model and k[2] in combine_years
            and k[1]!="all" and k[2] !="all"
            ]
    combo_hist = {}
    for k in combine_keys:
        if not combo_hist:
            hist_shape = hists[k]["state_hist"].shape
            combo_hist["state_hist"] = np.zeros(hist_shape, dtype=np.uint64)
            combo_hist["residual_hist"] = np.zeros(hist_shape, dtype=np.uint64)
            combo_hist["state_bounds"] = hists[k]["state_bounds"]
            combo_hist["residual_bounds"] = hists[k]["residual_bounds"]
            combo_hist["feats"] = hists[k]["feats"]
        combo_hist["state_hist"] += hists[k]["state_hist"]
        combo_hist["residual_hist"] += hists[k]["residual_hist"]
    hists[new_key] = combo_hist
    pkl.dump(hists, hists_pkl.open("wb"))

    ## combine static
    static = pkl.load(static_error_pkl.open("rb"))
    combine_keys = [k for k in static.keys()
            if k[3]==combine_model and k[2] in combine_years]
    combo_static = {}
    for k in combine_keys:
        stmp = static[k]
        ctmp = stmp["counts"][:,:,np.newaxis]
        if not combo_static:
            static_shape = stmp["err_state"].shape
            combo_static["err_state"] = np.zeros(static_shape)
            combo_static["err_residual"] = np.zeros(static_shape)
            combo_static["counts"] = np.zeros(
                    static_shape[:-1], dtype=np.uint64)
            combo_static["feats"] = stmp["feats"]
        combo_static["err_state"] += stmp["err_state"] * ctmp
        combo_static["err_residual"] += stmp["err_residual"] * ctmp
        combo_static["counts"] += stmp["counts"].astype(np.uint64)
    combo_static["err_state"] /= combo_static["counts"][:,:,np.newaxis]
    combo_static["err_residual"] /= combo_static["counts"][:,:,np.newaxis]
    m_zero = (combo_static["counts"] == 0)
    combo_static["err_state"][m_zero] = 0
    combo_static["err_residual"][m_zero] = 0
    static[new_key] = combo_static
    pkl.dump(static, static_error_pkl.open("wb"))

    ## combine horizons
    hor = pkl.load(error_horizons_pkl.open("rb"))
    combine_keys = [k for k in hor.keys()
            if k[3]==combine_model and k[2] in combine_years]
    combo_hor = {}
    for k in combine_keys:
        htmp = hor[k]
        if not combo_hor:
            hor_shape = htmp["state_avg"].shape
            combo_hor["state_avg"] = np.zeros(hor_shape)
            combo_hor["residual_avg"] = np.zeros(hor_shape)
            combo_hor["state_var"] = np.zeros(hor_shape)
            combo_hor["residual_var"] = np.zeros(hor_shape)
            combo_hor["counts"] = 0
            combo_hor["feats"] = htmp["feats"]
            combo_hor["pred_coarseness"] = htmp["pred_coarseness"]
        combo_hor["counts"] += htmp["counts"]
        combo_hor["state_avg"] += htmp["state_avg"] * htmp["counts"]
        combo_hor["residual_avg"] += htmp["residual_avg"] * htmp["counts"]
        combo_hor["state_var"] += htmp["state_var"] * htmp["counts"]
        combo_hor["residual_var"] += htmp["residual_var"] * htmp["counts"]
    combo_hor["state_avg"] /= combo_hor["counts"]
    combo_hor["residual_avg"] /= combo_hor["counts"]
    combo_hor["state_var"] /= combo_hor["counts"]
    combo_hor["residual_var"] /= combo_hor["counts"]
    hor[new_key] = combo_hor
    pkl.dump(hor, error_horizons_pkl.open("wb"))
    #'''
