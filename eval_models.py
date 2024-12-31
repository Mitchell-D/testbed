import numpy as np
import pickle as pkl
import random as rand
import json
import h5py
from datetime import datetime
from pathlib import Path
from pprint import pprint
from multiprocessing import Pool
import matplotlib.pyplot as plt
import tensorflow as tf
import gc

import model_methods as mm
import tracktrain as tt
import generators

output_conversions = {
        ## layerwise relative soil moisture in m^3/m^3
        "rsm-10":(
            ("soilm-10",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]/.1/1000-s[0])/(s[1]-s[0])",
            ),
        "rsm-40":(
            ("soilm-40",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]/.3/1000-s[0])/(s[1]-s[0])",
            ),
        "rsm-100":(
            ("soilm-100",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]/.6/1000-s[0])/(s[1]-s[0])",
            ),
        "rsm-200":(
            ("soilm-200",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]/1./1000-s[0])/(s[1]-s[0])",
            ),
        "rsm-fc":(
            ("soilm-fc",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]/2./1000-s[0])/(s[1]-s[0])",
            ),
        "soilm-10":(
            ("rsm-10",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]*(s[1]-s[0])+s[0])*1000*.1"
            ),
        "soilm-40":(
            ("rsm-40",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]*(s[1]-s[0])+s[0])*1000*.3"
            ),
        "soilm-100":(
            ("rsm-100",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]*(s[1]-s[0])+s[0])*1000*.6"
            ),
        "soilm-200":(
            ("rsm-200",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]*(s[1]-s[0])+s[0])*1000*1."
            ),
        "soilm-fc":(
            ("rsm-fc",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]*(s[1]-s[0])+s[0])*1000*2."
            ),
        }

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

def gen_sequence_predictions(
        model_dir:tt.ModelDir, sequence_generator_args:dict,
        weights_file_name:str=None, gen_batch_size=256, max_batches=None,
        dynamic_norm_coeffs:dict={}, static_norm_coeffs:dict={},
        gen_numpy=False, output_conversion=None, reset_model_each_batch=False):
    """
    Evaluates the provided model on a series of sequence files, and generates
    the predictions alongside the inputs

    :@param model_dir: tracktrain.ModelDir object for the desired model.
    :@param sequence_generator_args: Dict of arguments sufficient to initialize
        a sequence dataset generator with generators.sequence_dataset().
        The full dict should be json-serializable (ex no Path objects) because
        it will be stored alongside the prediction data as an attribute in
        order to init similar generators in the future.
    :@param weights_file_name: Name of the model weights file from the ModelDir
        directory to be used for inference.
    :@param gen_batch_size: Number of samples to draw from the gen at once.
    :@param max_batches: Maximum number of gen batches to store in the file.
    :@param dyanmic_norm_coeffs: Dictionary mapping feature names to 2-tuples
        (mean,stdev) representing linear norm coefficients for dynamic feats.
    :@param static_norm_coeffs: Dictionary mapping feature names to 2-tuples
        (mean,stdev) representing linear norm coefficients for static feats.
    :@param gen_numpy: If True, generate numpy arrays instead of tensors
    :@param output_conversion: Tragically high-level option to convert beteween
        relative soil moisture (rsm) and soil moisture area density (soilm)
        in model outputs and true values. If not None, output_conversion must
        be either "rsm_to_soilm" or "soilm_to_rsm".
    :@param reset_model_each_batch: Some large custom models seem to overflow
        session memory for some reason when evaluated on many large batches.
        This option will reset the tensorflow session state and reload the
        model weights for each batch if set to True.
    """
    ## Generator loop expects times since they will be recorded in the file.
    assert sequence_generator_args.get("yield_times") == True
    ## Make sure the initial normalization is handled by the generator
    if "dynamic_norm_coeffs" not in sequence_generator_args.keys():
        print(f"WARNING: generator doesn't have dynamic norm coefficients" + \
                ", so those provided to gen_sequence_predictions are assumed")
        sequence_generator_args["dynamic_norm_coeffs"] = dynamic_norm_coeffs
    if "static_norm_coeffs" not in sequence_generator_args.keys():
        print(f"WARNING: generator doesn't have static norm coefficients" + \
                ", so those provided to gen_sequence_predictions are assumed")
        sequence_generator_args["static_norm_coeffs"] = static_norm_coeffs

    ## load the model weights
    if not weights_file_name is None:
        weights_file_name = Path(weights_file_name).name
    print(f"Loading weights")
    model = model_dir.load_weights(weights_path=weights_file_name)

    ## prepare to convert output units if requested
    target_outputs = None
    if output_conversion == "rsm_to_soilm":
        target_outputs = [
                f.replace("rsm","soilm")
                for f in sequence_generator_args["pred_feats"]
                ]
        do_conversion = target_outputs != sequence_generator_args["pred_feats"]
    elif output_conversion == "soilm_to_rsm":
        target_outputs = [
                f.replace("soilm", "rsm")
                for f in sequence_generator_args["pred_feats"]
                ]
        do_conversion = target_outputs != sequence_generator_args["pred_feats"]
    else:
        do_conversion = False

    if do_conversion:
        sequence_generator_args["static_feats"] += ["wiltingp", "porosity"]
        p_idxs,p_derived,_ = generators._parse_feat_idxs(
                out_feats=target_outputs,
                src_feats=sequence_generator_args["pred_feats"],
                static_feats=["wiltingp", "porosity"],
                derived_feats=output_conversions,
                )

    print("Declaring generator")
    ## ignore any conditions restricting training
    ## (now not ignoring since the sequence gen doesn't come from model config)
    #sequence_generator_args["static_conditions"] = []
    gen = generators.sequence_dataset(**sequence_generator_args)

    ## collect normalization coefficients
    w_norm = np.array([
        tuple(dynamic_norm_coeffs[k])
        if k in dynamic_norm_coeffs.keys() else (0,1)
        for k in model_dir.config["feats"]["window_feats"]
        ])[np.newaxis,:]
    h_norm = np.array([
        tuple(dynamic_norm_coeffs[k])
        if k in dynamic_norm_coeffs.keys() else (0,1)
        for k in model_dir.config["feats"]["horizon_feats"]
        ])[np.newaxis,:]
    s_norm = np.array([
        tuple(static_norm_coeffs[k])
        if k in static_norm_coeffs.keys() else (0,1)
        for k in model_dir.config["feats"]["static_feats"]
        ])[np.newaxis,:]
    p_norm = np.array([
        tuple(dynamic_norm_coeffs[k])
        if k in dynamic_norm_coeffs.keys() else (0,1)
        for k in model_dir.config["feats"]["pred_feats"]
        ])[np.newaxis,:]


    ## Separate out norm coeffs for output conversion
    if do_conversion:
        convert_norm = s_norm[:,-2:,:]
        s_norm = s_norm[:,:-2,:]
    batch_counter = 0
    max_batches = (max_batches, -1)[max_batches is None]
    for (w,h,s,si,t),ys in gen.batch(gen_batch_size):
        print(f"Recieved new batch")
        if do_conversion:
            sparams = s[...,-2:]
            s = s[...,:-2]
            sparams = sparams * convert_norm[...,1] + convert_norm[...,0]

        if reset_model_each_batch:
            tf.keras.backend.clear_session()
            model = model_dir.load_weights(weights_path=weights_file_name)

        ## Normalize the predictions (assumes add_norm_layers not used!!!)
        print(f"Executing model")
        pr = model((w,h,s,si)) * p_norm[...,1]

        print(f"Calculating feature arrays")
        ## retain the initial observed state so the residual can be accumulated
        w = w * w_norm[...,1] + w_norm[...,0]
        h = h * h_norm[...,1] + h_norm[...,0]
        s = s * s_norm[...,1] + s_norm[...,0]
        ys = ys * p_norm[...,1] + p_norm[...,0]

        ## use the calculated functional parameters to convert units if needed
        if do_conversion:
            ps = tf.concat(
                    (ys[:,0,:][:,tf.newaxis,:],
                        (ys[:,0,:][:,tf.newaxis,:] + tf.cumsum(pr, axis=1))),
                    axis=1
                    )
            ps = generators._calc_feat_array(
                    src_array=ps,
                    static_array=sparams[:,np.newaxis],
                    stored_feat_idxs=tf.convert_to_tensor(p_idxs),
                    derived_data=p_derived,
                    )
            pr = ps[:,1:] - ps[:,:-1]
            ys = generators._calc_feat_array(
                    src_array=ys,
                    static_array=sparams[:,np.newaxis],
                    stored_feat_idxs=tf.convert_to_tensor(p_idxs),
                    derived_data=p_derived,
                    )

        th = t[:,-pr.shape[1]:]

        print(f"Generating result")
        if gen_numpy:
            w = w.numpy()
            h = h.numpy()
            s = s.numpy()
            si = si.numpy()
            th = th.numpy()
            ys = ys.numpy()
            pr = pr.numpy()
        batch_counter += 1
        if  batch_counter == max_batches:
            break
        yield (w,h,s,si,th),ys,pr

def sequence_preds_to_hdf5(model_dir:tt.ModelDir, sequence_generator_args:dict,
        pred_h5_path:Path, weights_file_name:str=None, chunk_size=256,
        gen_batch_size=256, max_batches=None,
        dynamic_norm_coeffs:dict={}, static_norm_coeffs:dict={}):
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
    :@param dynamic_norm_coeffs: Dictionary mapping feature names to 2-tuples
        (mean,stdev) representing linear norm coefficients for dynamic feats.
    :@param static_norm_coeffs: Dictionary mapping feature names to 2-tuples
        (mean,stdev) representing linear norm coefficients for static feats.
    """
    ## Initialize a prediction generator with the provided parameters
    gen = gen_sequence_predictions(
            model_dir=model_dir,
            sequence_generator_args=sequence_generator_args,
            weights_file_name=weights_file_name,
            gen_batch_size=gen_batch_size,
            max_batches=max_batches,
            dynamic_norm_coeffs=dynamic_norm_coeffs,
            static_norm_coeffs=static_norm_coeffs,
            )
    h5idx = 0
    batch_counter = 0
    max_batches = (max_batches, -1)[max_batches is None]
    F = None
    for i,((w,h,s,si,th),ys,pr) in enumerate(gen):
        if F is None:
            ## Create a new h5 file with datasets for the model (residual)
            ## predictions, timesteps, and initial states (last pred feats).
            F = h5py.File(
                    name=pred_h5_path,
                    mode="w-",
                    rdcc_nbytes=128*1024**2, ## use a 128MB cache
                    )
            output_shape = pr.shape[1:]
            P = F.create_dataset(
                    name="/data/preds",
                    shape=(0, *output_shape),
                    maxshape=(None, *output_shape),
                    chunks=(chunk_size, *output_shape),
                    compression="gzip",
                    )
            ## Times only include prediction horizon timestamps
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
            ## Store the generator arguments so the same kind can be re-init'd
            F["data"].attrs["gen_args"] = json.dumps(sequence_generator_args)
            F["data"].attrs["weights_file"] = Path(weights_file_name).name

        sample_slice = slice(h5idx, h5idx+ys.shape[0])
        h5idx += ys.shape[0]
        ## Only store initial state in the hdf5
        y0 = ys[:,0,:]
        ## Make room and dump the data to the file
        P.resize((h5idx, *pr.shape[1:]))
        T.resize((h5idx, th.shape[-1]))
        Y0.resize((h5idx, y0.shape[-1]))
        P[sample_slice,...] = pr.numpy()
        T[sample_slice,...] = th.numpy()
        Y0[sample_slice,...] = np.reshape(
                y0.numpy(),(y0.shape[0],y0.shape[-1]))
        F.flush()
        gc.collect()
        print(f"Loaded batch {i}; ({sample_slice.start}-{sample_slice.stop})")
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
    pass
