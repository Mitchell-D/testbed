
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
from list_feats import dynamic_coeffs,static_coeffs
from generators import gen_sequence_samples

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

def preds_to_hdf5(model_dir:tt.ModelDir, sequence_generator_args:dict,
        pred_h5_path:Path, weights_file_name:str=None, chunk_size=256,
        gen_batch_size=256, max_batches=None, pred_norm_coeffs:dict={}):
    """
    Evaluates the provided model on a series of sequence files, and stores the
    predictions in a new hdf5 file with a float32 dataset shaped (N,S,F_p) for
    N samples of length S sequences having F_p predicted features, and a uint
    dataset of epoch times shaped (N,S).

    :@param model_dir: tracktrain.ModelDir object for the desired model.
    :@param sequence_generator_args: Dict of arguments sufficient to initialize
        a sequence dataset generator with generators.gen_sequence_samples().
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

    gen = gen_sequence_samples(**sequence_generator_args)

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
        Y0[sample_slice,...] = np.squeeze(y0.numpy())
        #F.flush()

        batch_counter += 1
        if  batch_counter == max_batches:
            break
    F.close()
    return pred_h5_path

if __name__=="__main__":
    sequence_h5_dir = Path("data/sequences/")
    model_parent_dir = Path("data/models/new")
    pred_h5_dir = Path("data/predictions")

    model_name = "lstm-12"
    #weights_file = "lstm-7_095_0.283.weights.h5"
    #weights_file = "lstm-8_091_0.210.weights.h5"
    weights_file = None
    model_label = f"{model_name}-final"

    md = tt.ModelDir(
            model_parent_dir.joinpath(model_name),
            custom_model_builders={
                "lstm-s2s":lambda args:mm.get_lstm_s2s(**args),
                })

    ## Get a list of sequence hdf5s which will be independently evaluated
    seq_h5s = mm.get_seq_paths(
            sequence_h5_dir=sequence_h5_dir,
            region_strs=("ne", "nc", "nw", "se", "sc", "sw"),
            season_strs=("warm", "cold"),
            #time_strs=("2013-2018"),
            time_strs=("2018-2023"),
            )

    ## Ignore min,max values prepended to dynamic coefficients in list_feats
    dynamic_norm_coeffs={k:v[2:] for k,v in dynamic_coeffs}

    ## Arguments sufficient to initialize a generators.gen_sequence_samples
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
            }

    for h5_path in seq_h5s:
        seq_gen_args["sequence_hdf5s"] = [h5_path]
        _,region,season,time_range = Path(h5_path).stem.split("_")
        pred_h5_path = pred_h5_dir.joinpath(
                f"pred_{region}_{season}_{time_range}_{model_label}.h5")
        preds_to_hdf5(
                model_dir=md,
                sequence_generator_args=seq_gen_args,
                pred_h5_path=pred_h5_path,
                chunk_size=256,
                gen_batch_size=1024,
                weights_file_name=weights_file,
                pred_norm_coeffs=dynamic_norm_coeffs,
                )

    exit(0)

    for x,ys in gen.batch(1024):
        ## Normalize the predictions (assumes add_norm_layers not used!!!)
        pr = model(x) * p_norm[...,1] #+ p_norm[...,0]
        ys = ys * p_norm[...,1] + p_norm[...,0]

        ## Get the residual prediction from the model and accumulate it to
        ## the predicted state time series
        ps = ys[:,0,:][:,np.newaxis,:] + np.cumsum(pr, axis=1)
        ## Calculate the actual residual from labels
        yr = ys[:,1:]-ys[:,:-1]

        ## Calculate the error in the residual and state predictions
        es = ps - ys[:,1:,:]
        er = pr - yr
        break

    print(np.stack((pr,yr), axis=-1)[0])
    print()
    print(np.stack((ps,ys[:,1:]), axis=-1)[0])
    print()
    print(np.average(np.stack((pr,yr), axis=-1), axis=0))
    print()
    print(np.average(np.stack((ps,ys[:,1:]), axis=-1), axis=0))
    print()
    print(np.average(np.stack((er,es), axis=-1), axis=0))
