
from pathlib import Path
from random import random
import pickle as pkl

import numpy as np
import os
import sys

#import keras_tuner
import tensorflow as tf
from tensorflow.keras.layers import Layer,Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Concatenate, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, Flatten, RepeatVector
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Bidirectional
from tensorflow.keras.regularizers import L2
from tensorflow.keras import Input, Model

from list_feats import noahlsm_record_mapping, nldas_record_mapping
from model_methods import def_dense_kwargs, basic_dense, gen_hdf5_sample

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#'''
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
print(f"Tensorflow version: {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices())

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus):
    tf.config.experimental.set_memory_growth(gpus[0], True)
#'''
def make_generators(train_h5s, val_h5s, static_data,
        window_feat_idxs, horizon_feat_idxs, pred_feat_idxs,
        window, horizon, rand_seed=None, domain_mask=None):
    """
    Construct tf.data.Dataset from generators for the training and validation
    :@return: 2-tuple (training,validation) Datasets from mem-mapped hdf5s
    """
    ## nested output signature for gen_hdf5_sample
    out_sig = ({
        "window":tf.TensorSpec(
            shape=(window,len(window_feat_idxs)), dtype=tf.float16),
        "horizon":tf.TensorSpec(
            shape=(horizon,len(horizon_feat_idxs)), dtype=tf.float16),
        "static":tf.TensorSpec(
            shape=(static_data.shape[-1],), dtype=tf.float16)
        },
        tf.TensorSpec(shape=(horizon,len(pred_feat_idxs)), dtype=tf.float16))

    gT = tf.data.Dataset.from_generator(
            gen_hdf5_sample,
            args=(
                train_h5s,
                static_data,
                window,
                horizon,
                window_feat_idxs,
                horizon_feat_idxs,
                pred_feat_idxs,
                rand_seed,
                domain_mask,
                ),
            output_signature=out_sig
            )

    gV = tf.data.Dataset.from_generator(
            gen_hdf5_sample,
            args=(
                val_h5s,
                static_data,
                window,
                horizon,
                window_feat_idxs,
                horizon_feat_idxs,
                pred_feat_idxs,
                rand_seed,
                domain_mask,
                ),
            output_signature=out_sig
            )
    return gT,gV

if __name__=="__main__":
    """ Directory with sub-directories for each model. """
    data_dir = Path("data")
    model_parent_dir = data_dir.joinpath("models")

    """ Paths to relevant data """
    static_path = data_dir.joinpath("nldas2_static_all.pkl")

    """
    Training configuration for a single model run.
    """
    ## Identifying label for this model
    model_name= "ff_0"
    ## Size of batches in samples
    batch_size = 32
    ## Batches to draw asynchronously from the generator
    batch_buffer = 4
    ## Seed for subsampling training and validation data
    rand_seed = 20231121

    """ Labels for each feature category, in the order they are provided """
    window_feats = [
            "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
            "dlwrf", "ncrain", "cape", "pevap", "apcp", "dswrf",
            "soilm-10", "soilm-40", "soilm-100", "soilm-200"]
    horizon_feats = [
            "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
            "dlwrf", "ncrain", "cape", "pevap", "apcp", "dswrf"]
    pred_feats = ['soilm-10', 'soilm-40', 'soilm-100', 'soilm-200']

    """ Load information from static data """
    static_dict = pkl.load(static_path.open("rb"))
    static = static_dict["soil_comp"]
    lon,lat = static_dict["geo"]

    """ Construct a geographic mask setting valid data points to True"""
    ## Geographically constrain to the South East
    m_lon = np.logical_and(-100<=lon,lon<=-80)
    m_lat = np.logical_and(30<=lat,lat<=40)
    ## Don't consider water or urban surfaces
    m_water = static_dict["soil_type_ints"] == 0
    m_urban = static_dict["soil_type_ints"] == 13
    m_sfc = np.logical_not(np.logical_or(m_water, m_urban))
    m_geo = np.logical_and(m_lon, m_lat)
    ## Make a (lat,lon) shaped bool mask for the data generators
    m_valid = np.logical_and(m_sfc, m_geo)

    ## Make the directory for this model run, ensuring no name collision.
    model_dir = model_parent_dir.joinpath(model_name)
    #assert not model_dir.exists()
    #model_dir.mkdir()

    ## Define callbacks for model progress tracking
    model = basic_dense(
            name="ff", ## feedforward
            node_list=[64,64,64,48,32,24,16],
            window_feats=len(window_feats),
            horizon_feats=len(horizon_feats),
            static_feats=static.shape[-1],
            pred_feats=len(pred_feats),
            batchnorm=True,
            dropout_rate=0.2,
            dense_kwargs={}
            )

    ## Write a model summary to stdout and to a file
    model.summary()

    with model_dir.joinpath(model_name+"_summary.txt").open("w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    _,feat_order = zip(*nldas_record_mapping, *noahlsm_record_mapping)
    gT,gV = make_generators(
            train_h5s=[data_dir.joinpath(f"feats/feats_{y}.hdf5").as_posix()
                for y in [2015,2016,2018,2019,2021]],
            val_h5s=[data_dir.joinpath(f"feats/feats_{y}.hdf5").as_posix()
                for y in [2017,2020]],
            static_data=static,
            window_feat_idxs=[feat_order.index(f) for f in window_feats],
            horizon_feat_idxs=[feat_order.index(f) for f in horizon_feats],
            pred_feat_idxs=[feat_order.index(f) for f in pred_feats],
            window=1,
            horizon=1,
            rand_seed=rand_seed,
            domain_mask=m_valid
            )

    model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="mse",
            metrics=["mse"],
            )

    """ Create callbacks for training diagnostics"""
    c_early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=50)
    c_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            monitor="val_loss", save_best_only=True,
            filepath=model_dir.joinpath(
                model_name+"_{epoch}_{mse:.2f}.hdf5"))
    c_csvlog = tf.keras.callbacks.CSVLogger(
            model_dir.joinpath("prog.csv"))


    print(f"Fitting model")
    ## Train the model on the generated tensors
    hist = model.fit(
            gT.batch(batch_size).prefetch(batch_buffer),
            epochs=1000,
            ## Number of batches to draw per epoch. Use full dataset by default
            steps_per_epoch=100, ## 3,200 samples per epoch
            validation_data=gV.batch(batch_size).prefetch(batch_buffer),
            ## batches of validation data to draw per epoch
            validation_steps=25, ## 3,200 samples per validation
            validation_freq=1, ## Report validation loss each epoch
            callbacks=[
                c_early_stop,
                c_checkpoint,
                c_csvlog,
               ],
            verbose=2,
            )
    model.save(model_dir.joinpath(model_name+".keras"))
