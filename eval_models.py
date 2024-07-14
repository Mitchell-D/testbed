
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

def get_seq_paths(sequence_h5_dir:Path,
        region_strs:list=[], season_strs:list=[], time_strs:list=[]):
    """
    Constrain sequences hdf5 files by their underscore-separated fields
    """
    return tuple([
        str(p) for p in sequence_h5_dir.iterdir()
        if len(p.stem.split("_")) == 4 and all((
            p.stem.split("_")[0] == "sequences",
            p.stem.split("_")[1] in region_strs,
            p.stem.split("_")[2] in season_strs,
            p.stem.split("_")[3] in time_strs,
            ))
        ])

def add_norm_layers(md:tt.ModelDir, weights_file:str=None,
        dynamic_norm_coeffs:dict={}, static_norm_coeffs:dict={},
        save_new_model=False):
    """ """
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
    model_name = "test-2"


    md = tt.ModelDir(
            model_parent_dir.joinpath(model_name),
            custom_model_builders={
                "lstm-s2s":lambda args:mm.get_lstm_s2s(**args),
                })

    model = add_norm_layers(
            md=md,
            dynamic_norm_coeffs={k:v[2:] for k,v in dynamic_coeffs},
            static_norm_coeffs=dict(static_coeffs),
            )

    #model.summary(expand_nested=True)

    seq_h5s = get_seq_paths(
            sequence_h5_dir=sequence_h5_dir,
            region_strs=("ne", "nc", "nw", "se", "sc", "sw"),
            season_strs=("warm"),
            time_strs=("2018-2023"),
            )

    gen = get_seq_gen(
            md=md,
            sequence_hdf5s=seq_h5s,
            num_procs=6,
            frequency=8,
            sample_on_frequency=True,
            )

    for x,y in gen.batch(64).prefetch(2):
        p = model(x)
        break
    print(p.shape, y.shape)
