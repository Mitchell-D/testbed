""" """
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
from generators import gen_timegrid_samples,gen_sequence_samples
from list_feats import nldas_record_mapping,noahlsm_record_mapping
from list_feats import umd_veg_classes, statsgo_textures
from list_feats import dynamic_coeffs,static_coeffs


config = {
        "feats":{
            "window_feats":[
                "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
                "dlwrf", "dswrf", "apcp",
                "soilm-10", "soilm-40", "soilm-100", "soilm-200", "weasd" ],
            "horizon_feats":[
                "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
                "dlwrf", "dswrf", "apcp" ],
            "pred_feats":[
                "soilm-10", "soilm-40", "soilm-100", "soilm-200", "weasd"],
            "static_feats":[
                "pct_sand", "pct_silt", "pct_clay", "elev", "elev_std"],
            "static_int_feats":["int_veg"],
            "total_static_int_input_size":14,
            },

        "model":{
            "window_size":24,
            "horizon_size":24*14,
            "input_lstm_depth_nodes":[32,32,64],
            "output_lstm_depth_nodes":[32,32,64],
            "static_int_embed_size":3,
            "input_linear_embed_size":32,
            "bidirectional":True,

            "batchnorm":True,
            "dropout_rate":0.,
            "input_lstm_kwargs":{},
            "output_lstm_kwargs":{},
            },

        ## Exclusive to compile_and_build_dir
        "compile":{
            "learning_rate":1e-3,
            "loss":"res_only",
            "metrics":["res_only"],#["mse", "mae"],
            },

        ## Exclusive to train
        "train":{
            ## metric evaluated for stagnation
            "early_stop_metric":"val_residual_loss",
            "early_stop_patience":64, ## number of epochs before stopping
            "save_weights_only":True,
            "batch_size":64,
            "batch_buffer":2,
            "max_epochs":256, ## maximum number of epochs to train
            "val_frequency":1, ## epochs between validations
            "steps_per_epoch":128, ## batches to draw per epoch
            "validation_steps":64, ## batches to draw per validation
            },

        ## Exclusive to generator init
        "data":{
            "train_files":None,
            "train_procs":6,

            "val_files":None,
            "val_procs":4,

            "frequency":3,
            "block_size":64,
            "buf_size_mb":512,
            "deterministic":False,
            },

        "model_name":"test-2",
        "model_type":"lstm-s2s",
        "seed":200007221750,
        "notes":"",
        }

if __name__=="__main__":
    sequences_dir = Path("/rstor/mdodson/thesis/sequences")
    model_parent_dir = Path("data/models/new")

    '''
    config["data"]["train_files"] = tuple(map(str,[
            sequences_dir.joinpath(f"sequences_loam_se.h5")]))
    config["data"]["val_files"] = tuple(map(str,[
            sequences_dir.joinpath(f"sequences_loam_se.h5")]))
    config["data"]["train_files"] = tuple(map(str,[
        p for p in sequences_dir.iterdir()
        if "se_warm_2013" in p.stem
        ]))
    '''

    region_strs = ("_se_", "_sc_", "_ne_")
    time_strs = ("2013",)
    season_strs = ("_warm_",)

    config["data"]["train_files"] = val_files = tuple([
            str(p) for p in sequences_dir.iterdir()
            if all(map(
                lambda t:any(s in p.stem for s in t),
                (region_strs, time_strs, season_strs)
                ))
            ])
    config["data"]["val_files"] = val_files = tuple([
            str(p) for p in sequences_dir.iterdir()
            if all(map(
                lambda t:any(s in p.stem for s in t),
                (region_strs, time_strs, season_strs)
                ))
            ])


    """ Declare training and validation dataset generators using the config """
    data_t = gen_sequence_samples(
            sequence_hdf5s=config["data"]["train_files"],
            num_procs=config["data"]["train_procs"],
            sample_on_frequency=False,
            dynamic_norm_coeffs={k:v[2:] for k,v in dynamic_coeffs},
            static_norm_coeffs=dict(static_coeffs),
            seed=config["seed"],
            **config["feats"],
            **config["data"],
            )
    data_v = gen_sequence_samples(
            sequence_hdf5s=config["data"]["train_files"],
            num_procs=config["data"]["val_procs"],
            sample_on_frequency=True,
            dynamic_norm_coeffs={k:v[2:] for k,v in dynamic_coeffs},
            static_norm_coeffs=dict(static_coeffs),
            seed=config["seed"],
            **config["feats"],
            **config["data"],
            )

    '''
    for (w,h,s,si),p in data_t.batch(64):
        print(np.average(w, axis=(0,1)))
        print(np.average(h, axis=(0,1)))
        print(np.average(p, axis=(0,1)))

    for (w,h,s,si),p in data_v.batch(64):
        print(np.average(w, axis=(0,1)))
        print(np.average(h, axis=(0,1)))
        print(np.average(p, axis=(0,1)))


        exit(0)
    '''

    """ Initialize a custom residual loss function """
    res_loss = mm.get_residual_loss_fn(
            residual_ratio=.6,
            use_mse=False,
            )
    res_only = mm.get_residual_loss_fn(
            residual_ratio=1.,
            use_mse=False,
            )

    """ Update model configuration with feature vector size information """
    config["model"].update({
        "num_window_feats":len(config["feats"]["window_feats"]),
        "num_horizon_feats":len(config["feats"]["horizon_feats"]),
        "num_static_feats":len(config["feats"]["static_feats"]),
        "num_static_int_feats":config["feats"]["total_static_int_input_size"],
        "num_pred_feats":len(config["feats"]["pred_feats"]),
        })

    """ Initialize the model and build its directory """
    model,md = tt.ModelDir.build_from_config(
            config=config,
            model_parent_dir=model_parent_dir,
            print_summary=True,
            custom_model_builders={
                "lstm-s2s":lambda args:mm.get_lstm_s2s(**args),
                },
            custom_losses={ "res_loss":res_loss, "res_only":res_only},
            custom_metrics={ "res_only":res_only },
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
        )