from pathlib import Path
from random import random
import pickle as pkl
import json
import numpy as np
import os
import sys

#import keras_tuner
import tensorflow as tf
from tensorflow.keras.layers import Layer,Masking,Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Concatenate, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, Flatten, RepeatVector
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Bidirectional
from tensorflow.keras.regularizers import L2
from tensorflow.keras import Input, Model

from list_feats import noahlsm_record_mapping, nldas_record_mapping
#from model_methods import def_dense_kwargs, basic_dense, gen_hdf5_sample
import model_methods as mm

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

if __name__=="__main__":
    """ Directory with sub-directories for each model. """
    data_dir = Path("/rstor/mdodson/thesis/")
    model_parent_dir = Path("/rhome/mdodson/testbed/data/models-seus")

    config = {
            "model_name":"tcn-seus-0",
            "batch_size":256,
            "batch_buffer":4,
            "window_feats":[
                "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
                "dlwrf", "ncrain", "cape", "pevap", "apcp", "dswrf",
                "soilm-10", "soilm-40", "soilm-100", "soilm-200"],
            "horizon_feats":[
                "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
                "dlwrf", "ncrain", "cape", "pevap", "apcp", "dswrf"],
            "pred_feats":['soilm-10', 'soilm-40', 'soilm-100', 'soilm-200'],
            "static_feats":[
                "pct_sand", "pct_silt", "pct_clay", "elev", "elev_std"],
            "window_and_horizon_size":12,
            "dense_units":256,
            "dilation_layers":[2,4,6],
            "num_filters":64,
            "kernel_size":4,
            "train_h5s":[data_dir.joinpath(f"shuffle_SEUS_{y}.h5").as_posix()
                for y in [2015,2017,2019,2021]],
            "val_h5s":[data_dir.joinpath(f"shuffle_SEUS_{y}.h5").as_posix()
                for y in [2018,2020]],
            "loss":"mse",
            "metrics":["mse", "mae"],
            "early_stop_patience":20, ## number of epochs before stopping
            "max_epochs":2048, ## maximum number of epochs to train
            "train_steps_per_epoch":100, ## number of batches per epoch
            "val_steps_per_epoch":32, ## number of batches per validation
            "val_frequency":1, ## epochs between validation
            "learning_rate":1e-2,
            "notes":"Same as tcn-1 except only trained on SEUS pixels",
            }

    ## Make the directory for this model run, ensuring no name collision.
    #'''
    model_dir = model_parent_dir.joinpath(config["model_name"])
    assert not model_dir.exists()
    model_dir.mkdir()
    model_json_path = model_dir.joinpath(f"{config['model_name']}_config.json")
    model_json_path.open("w").write(json.dumps(config,indent=4))
    #'''
    ## Define callbacks for model progress tracking
    model = mm.temporal_convolution(
            window_and_horizon_size=config["window_and_horizon_size"],
            num_window_feats=len(config["window_feats"]),
            num_horizon_feats=len(config["horizon_feats"]),
            num_static_feats=len(config["static_feats"]),
            num_pred_feats=len(config["pred_feats"]),
            dense_units=config["dense_units"],
            dilation_layers=config["dilation_layers"],
            num_filters=config["num_filters"],
            kernel_size=config["kernel_size"],
            )

    ## Write a model summary to stdout and to a file
    model.summary()

    summary_path = model_dir.joinpath(config["model_name"]+"_summary.txt")
    with summary_path.open("w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    gT,gV = mm.get_sample_generator(
            train_h5s=config["train_h5s"],
            val_h5s=config["val_h5s"],
            window_size=config["window_and_horizon_size"],
            horizon_size=config["window_and_horizon_size"],
            window_feats=config["window_feats"],
            horizon_feats=config["horizon_feats"],
            pred_feats=config["pred_feats"],
            static_feats=config["static_feats"],
            )

    model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config["learning_rate"]),
            loss=config["loss"],
            metrics=config["metrics"],
            )

    """ Create callbacks for training diagnostics"""
    c_early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config["early_stop_patience"])
    c_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            monitor="val_loss", save_best_only=True,
            filepath=model_dir.joinpath(
                config['model_name']+"_{epoch:03}_{val_mae:.2f}.hdf5"))
    c_csvlog = tf.keras.callbacks.CSVLogger(
            model_dir.joinpath(f"{config['model_name']}_prog.csv"))


    ## Train the model on the generated tensors
    hist = model.fit(
            gT.batch(config["batch_size"]).prefetch(config["batch_buffer"]),
            epochs=config["max_epochs"],
            ## Number of batches to draw per epoch. Use full dataset by default
            steps_per_epoch=config["train_steps_per_epoch"],
            validation_data=gV.batch(
                config["batch_size"]).prefetch(config["batch_buffer"]),
            ## batches of validation data to draw per epoch
            validation_steps=config["val_steps_per_epoch"],
            ## Number of epochs to wait between validation runs.
            validation_freq=config["val_frequency"],
            callbacks=[
                c_early_stop,
                c_checkpoint,
                c_csvlog,
               ],
            verbose=2,
            )
    model.save(model_dir.joinpath(config["model_name"]+".keras"))
