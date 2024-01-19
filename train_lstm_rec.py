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

def get_lstm_rec(window_size, num_window_feats, num_horizon_feats,
        num_static_feats, num_pred_feats, input_lstm_depth_nodes,
        output_dense_nodes, input_dense_nodes=None, bidirectional=True,
        batchnorm=True, dropout_rate=0.0, lstm_kwargs={}, dense_kwargs={}):
    """
    Sequence -> Vector network with a LSTM window encoder and a dense layer
    stack for next-step prediction
    """
    w_in = Input(shape=(window_size,num_window_feats,), name="in_window")
    h_in = Input(shape=(1,num_horizon_feats,), name="in_horizon")
    s_in = Input(shape=(num_static_feats,), name="in_static")
    s_seq = RepeatVector(window_size)(s_in)
    seq_in = Concatenate(axis=-1)([w_in,s_seq])

    prev_layer = seq_in
    if not input_dense_nodes is None:
        prev_layer = TimeDistributed(Dense(input_dense_nodes))(prev_layer)

    ## Get a LSTM stack that accepts a (horizon,feats) sequence and outputs
    ## a single vector
    prev_layer = mm.get_lstm_stack(
            name="enc_lstm",
            layer_input=prev_layer,
            node_list=input_lstm_depth_nodes,
            return_seq=False,
            bidirectional=bidirectional,
            lstm_kwargs=lstm_kwargs,
            )
    ## Concatenate the encoder output with the horizon data
    prev_layer = Concatenate(axis=-1)([
        prev_layer, Reshape(target_shape=(num_horizon_feats,))(h_in)
        ])
    prev_layer = mm.get_dense_stack(
            name="dec_dense",
            node_list=output_dense_nodes,
            layer_input=prev_layer,
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
            dense_kwargs=dense_kwargs,
            )

    inputs = {"window":w_in,"horizon":h_in,"static":s_in}
    ## Reshape output to match the data tensor
    output = Reshape(target_shape=(1,num_pred_feats))(
            Dense(num_pred_feats)(prev_layer))
    return Model(inputs=inputs, outputs=[output])

if __name__=="__main__":
    """ Directory with sub-directories for each model. """
    data_dir = Path("/rstor/mdodson/thesis/")
    model_parent_dir = Path("/rhome/mdodson/testbed/data/models-seus")

    config = {
            "model_name":"lstm-rec-seus-0",
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
            "window_size":12,
            "horizon_size":1,
            "dropout_rate":.1,
            "batchnorm":True,
            "bidirectional":False,
            "input_lstm_kwargs":{},
            "output_lstm_kwargs":{},
            "input_lstm_depth_nodes":[128,96,64,32,16],
            "output_dense_nodes":[1024,1024,512,512,256,256,192,192,128,128],
            "input_dense_nodes":128,
            #"input_dense_nodes":None,
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
            "learning_rate":1e-4,
            "notes":"Same as lstm-rec-1 except trained on SEUS pixels only",
            }

    ## Make the directory for this model run, ensuring no name collision.
    model_dir = model_parent_dir.joinpath(config["model_name"])
    assert not model_dir.exists()
    model_dir.mkdir()
    model_json_path = model_dir.joinpath(f"{config['model_name']}_config.json")
    model_json_path.open("w").write(json.dumps(config,indent=4))

    ## Define callbacks for model progress tracking
    model = get_lstm_rec(
            window_size=config["window_size"],
            num_window_feats=len(config["window_feats"]),
            num_horizon_feats=len(config["horizon_feats"]),
            num_static_feats=len(config["static_feats"]),
            num_pred_feats=len(config["pred_feats"]),
            input_lstm_depth_nodes=config["input_lstm_depth_nodes"],
            output_dense_nodes=config["output_dense_nodes"],
            input_dense_nodes=config["input_dense_nodes"],
            bidirectional=config["bidirectional"],
            batchnorm=config["batchnorm"],
            dropout_rate=config["dropout_rate"],
            lstm_kwargs=config["input_lstm_kwargs"],
            dense_kwargs=config["output_lstm_kwargs"]
            )

    ## Write a model summary to stdout and to a file
    model.summary()

    summary_path = model_dir.joinpath(config["model_name"]+"_summary.txt")
    with summary_path.open("w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    gT,gV = mm.get_sample_generator(
            train_h5s=config["train_h5s"],
            val_h5s=config["val_h5s"],
            window_size=config["window_size"],
            horizon_size=config["horizon_size"],
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
