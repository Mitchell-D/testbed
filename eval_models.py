
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

def get_seq_gen(md:tt.ModelDir, sequence_hdf5s:list,
        dynamic_norm_coeffs={}, static_norm_coeffs={},
        num_procs=1, frequency=1, sample_on_frequency=True, seed=None):
    """
    """
    return gen_sequence_samples(
        sequence_hdf5s=sequence_hdf5s,
        num_procs=num_procs,
        frequency=frequency,
        sample_on_frequency=sample_on_frequency,
        dynamic_norm_coeffs=dynamic_norm_coeffs,
        static_norm_coeffs=static_norm_coeffs,
        seed=seed,
        **md.config["feats"],
        )

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

if __name__=="__main__":
    sequence_h5_dir = Path("data/sequences/")
    model_parent_dir = Path("data/models/new")
    model_name = "lstm-4"


    md = tt.ModelDir(
            model_parent_dir.joinpath(model_name),
            custom_model_builders={
                "lstm-s2s":lambda args:mm.get_lstm_s2s(**args),
                })
    model = md.load_weights(weights_path="lstm-4_009_0.028.weights.h5")

    '''
    model = add_norm_layers(
            md=md,
            dynamic_norm_coeffs={k:v[2:] for k,v in dynamic_coeffs},
            static_norm_coeffs=dict(static_coeffs),
            )
    '''

    seq_h5s = mm.get_seq_paths(
            sequence_h5_dir=sequence_h5_dir,
            region_strs=("ne", "nc", "nw", "se", "sc", "sw"),
            season_strs=("warm"),
            #time_strs=("2018-2023"),
            time_strs=("2013-2018"),
            )

    dynamic_norm_coeffs={k:v[2:] for k,v in dynamic_coeffs}
    p_norm = np.array([
        tuple(dynamic_norm_coeffs[k])
        if k in dynamic_norm_coeffs.keys() else (0,1)
        for k in md.config["feats"]["pred_feats"]
        ])[np.newaxis,:]

    gen = get_seq_gen(
            md=md,
            sequence_hdf5s=seq_h5s,
            num_procs=6,
            frequency=1,
            sample_on_frequency=True,
            dynamic_norm_coeffs=dynamic_norm_coeffs,
            static_norm_coeffs=dict(static_coeffs),
            )

    for x,ys in gen.batch(64).prefetch(2):
        ## Normalize the predictions (assumes add_norm_layers not used!!!)
        pr = model(x) * p_norm[...,1] + p_norm[...,0]
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
