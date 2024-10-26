"""
Script for dispatching single tracktrain-based training runs at a time,
each based on the configuration dict below. Each run will create a new
ModelDir directory and populate it with model info, the configuration,
and intermittent models saved duing training.
"""
import numpy as np
import pickle as pkl
import random as rand
import json
import h5py
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
import matplotlib.pyplot as plt

import tracktrain as tt

import model_methods as mm
from generators import sequence_dataset
from list_feats import dynamic_coeffs,static_coeffs,derived_feats

import tensorflow as tf
print(tf.config.list_physical_devices())
print("GPUs: ", len(tf.config.list_physical_devices('GPU')))

config = {
        "feats":{
            "window_feats":[
                "lai", "veg", "tmp", "spfh", "pres","ugrd", "vgrd",
                "dlwrf", "dswrf", "apcp",
                "rsm-10", "rsm-40", "rsm-100"],
                #"soilm-10", "soilm-40", "soilm-100", "soilm-200", "weasd" ],
                #"weasd" ],
            "horizon_feats":[
                "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
                "dlwrf", "dswrf", "apcp", "weasd"],
            "pred_feats":[
                #"soilm-10", "soilm-40", "soilm-100", "soilm-200", "weasd"],
                "rsm-10", "rsm-40", "rsm-100"],
                #"rsm-fc"],
            "static_feats":[
                "pct_sand", "pct_silt", "pct_clay", "elev", "elev_std"],
                #"elev", "elev_std"],
            "static_int_feats":["int_veg"],
            "total_static_int_input_size":14,
            "pred_coarseness":1,
            },

        "model":{
            "window_size":24,
            "horizon_size":24*14,
            "input_lstm_depth_nodes":[48,48,48,48,48],
            "output_lstm_depth_nodes":[256,256,256,256,256],
            "static_int_embed_size":4,
            "input_linear_embed_size":32,
            "bidirectional":False,

            "batchnorm":False,
            "dropout_rate":0.05,
            "input_lstm_kwargs":{},
            "output_lstm_kwargs":{},
            "bias_state_rescale":True,
            },

        ## Exclusive to compile_and_build_dir
        "compile":{
            "optimizer":"adam",
            "learning_rate":5e-2,
            "loss":"res_loss",
            #"loss":"snow_loss",
            #"metrics":["res_only"],#["mse", "mae"],
            "metrics":["res_only", "state_only"],#["mse", "mae"],
            },

        ## Exclusive to train
        "train":{
            ## metric evaluated for stagnation
            #"early_stop_metric":"val_residual_loss",
            "early_stop_metric":"val_loss",
            "early_stop_patience":48, ## number of epochs before stopping
            "save_weights_only":True,
            "batch_size":32,
            "batch_buffer":10,
            "max_epochs":1024, ## maximum number of epochs to train
            "val_frequency":1, ## epochs between validations
            "steps_per_epoch":256, ## batches to draw per epoch
            "validation_steps":128, ## batches to draw per validation
            "repeat_data":True,
            "lr_scheduler":"cyclical",
            "lr_scheduler_args":{
                "lr_min":1e-5,
                "lr_max":1e-2,
                "inc_epochs":2,
                "dec_epochs":6,
                "decay":.01,
                "log_scale":True,
                },
            },

        ## Exclusive to generator init
        "data":{
            "train_files":None,
            "train_procs":8,

            "val_files":None,
            "val_procs":6,

            "frequency":3, ## determines training/validation balance
            "block_size":8,
            "buf_size_mb":1024,
            "deterministic":False,

            "train_region_strs":("se", "sc", "sw", "ne", "nc", "nw"),
            #"train_region_strs":("se",),
            "train_time_strs":("2012-2015", "2015-2018"),
            "train_season_strs":("warm","cold"),
            #"train_season_strs":("warm",),
            #"train_season_strs":("cold",),

            "val_region_strs":("se", "sc", "sw", "ne", "nc", "nw"),
            #"val_region_strs":("se",),
            "val_time_strs":("2012-2015", "2015-2018"),
            "val_season_strs":("warm","cold"),
            #"val_season_strs":("warm",),
            #"val_season_strs":("cold",),

            "static_conditions":[
                ## select soil indeces
                #(("int_soil",), "lambda s:np.any(np.stack([s[0]==v " "for v in (1,2,3,7,10)], axis=-1), axis=-1)"),
                ## subset by percent sand
                #(("pct_sand",), "lambda s:s[0]>.55"),
                ],

            "loss_fn_args":{
                "residual_ratio":1.,
                "use_mse":False,
                "residual_norm":None, ## this value set below
                "residual_magnitude_bias":10,
                }
            },

        "model_name":"lstm-rsm-10",
        "model_type":"lstm-s2s",
        "seed":200007221750,
        "notes":"same as rsm-9 except way wider (256 node) and 5-layer model",
        }

if __name__=="__main__":
    sequences_dir = Path("/rstor/mdodson/thesis/sequences")
    model_parent_dir = Path("data/models/new")


    """ Specify the training and validation files """
    config["data"]["train_files"] = mm.get_seq_paths(
            sequence_h5_dir=sequences_dir,
            region_strs=config["data"]["train_region_strs"],
            season_strs=config["data"]["train_season_strs"],
            time_strs=config["data"]["train_time_strs"],
            )

    config["data"]["val_files"] = mm.get_seq_paths(
            sequence_h5_dir=sequences_dir,
            region_strs=config["data"]["val_region_strs"],
            season_strs=config["data"]["val_season_strs"],
            time_strs=config["data"]["val_time_strs"],
            )

    """ Declare training and validation dataset generators using the config """
    data_t = sequence_dataset(
            sequence_hdf5s=config["data"]["train_files"],
            num_procs=config["data"]["train_procs"],
            sample_on_frequency=False,
            dynamic_norm_coeffs={k:v[2:] for k,v in dynamic_coeffs},
            static_norm_coeffs=dict(static_coeffs),
            derived_feats=derived_feats,
            seed=config["seed"],
            shuffle=True,
            **config["feats"],
            **config["data"],
            )
    data_v = sequence_dataset(
            sequence_hdf5s=config["data"]["val_files"],
            num_procs=config["data"]["val_procs"],
            sample_on_frequency=True,
            dynamic_norm_coeffs={k:v[2:] for k,v in dynamic_coeffs},
            static_norm_coeffs=dict(static_coeffs),
            derived_feats=derived_feats,
            seed=config["seed"],
            shuffle=True,
            **config["feats"],
            **config["data"],
            )

    """ Get residual norm coeffs from the residual standard deviations """
    config["data"]["residual_norm"] = [
            dict(dynamic_coeffs)["res_"+l][-1]
            for l in config["feats"]["pred_feats"]
            ]

    """ Initialize a custom residual loss function """
    res_loss = mm.get_residual_loss_fn(
            **config["data"]["loss_fn_args"],
            fn_name="res_loss"
            )
    res_only = mm.get_residual_loss_fn(
            residual_ratio=1.,
            use_mse=config["data"]["loss_fn_args"]["use_mse"],
            residual_norm=config["data"]["loss_fn_args"].get("residual_norm"),
            fn_name="res_only",
            )
    state_only = mm.get_residual_loss_fn(
            residual_ratio=0.,
            use_mse=config["data"]["loss_fn_args"]["use_mse"],
            residual_norm=config["data"]["loss_fn_args"].get("residual_norm"),
            fn_name="state_only",
            )
    """ Initialize snow loss function """
    rmb = config["data"]["loss_fn_args"]["residual_magnitude_bias"]
    snow_loss = mm.get_snow_loss_fn(
            zero_point=[
                -1 * dict(dynamic_coeffs)[k][2] / dict(dynamic_coeffs)[k][3]
                for k in config["feats"]["pred_feats"]
                ],
            use_mse=config["data"]["loss_fn_args"]["use_mse"],
            residual_norm=config["data"]["loss_fn_args"].get("residual_norm"),
            residual_magnitude_bias=rmb,
            )


    """ Update model configuration with feature vector size information """
    config["model"].update({
        "num_window_feats":len(config["feats"]["window_feats"]),
        "num_horizon_feats":len(config["feats"]["horizon_feats"]),
        "num_static_feats":len(config["feats"]["static_feats"]),
        "num_static_int_feats":config["feats"]["total_static_int_input_size"],
        "num_pred_feats":len(config["feats"]["pred_feats"]),
        "pred_coarseness":config["feats"]["pred_coarseness"],
        })

    """ Initialize the model and build its directory """
    model,md = tt.ModelDir.build_from_config(
            config=config,
            model_parent_dir=model_parent_dir,
            print_summary=True,
            custom_model_builders={
                "lstm-s2s":lambda args:mm.get_lstm_s2s(**args),
                },
            custom_losses={"res_loss":res_loss, "snow_loss":snow_loss},
            custom_metrics={"res_only":res_only,"state_only":state_only},
            )

    #'''
    """ optionally generate an image model diagram ; has `pydot` dependency """
    from keras.utils import plot_model
    plot_model(model, to_file=md.dir.joinpath(f"{md.name}.png"),
               show_shapes=True, show_layer_names=True, expand_nested=True,
               show_layer_activations=True)
    #'''

    """
    Train the model. Expects the following fields to be in config:
    "early_stop_metric","early_stop_patience","save_weights_only",
    "batch_size","batch_buffer","max_epochs","val_frequency",
    """
    best_model = tt.train(
        model_dir_path=md.dir,
        train_config=config["train"],
        compiled_model=model,
        gen_training=data_t,
        gen_validation=data_v,
        custom_lr_schedulers={
            "cyclical":mm.get_cyclical_lr(
                **config["train"].get("lr_scheduler_args", {})),
            },
        )
