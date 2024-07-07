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

from generators import gen_timegrid_samples,gen_sequence_samples
from list_feats import nldas_record_mapping,noahlsm_record_mapping
from list_feats import umd_veg_classes, statsgo_textures


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
            },

        "model":{
            "model_type":"ff",
            },

        ## Exclusive to compile_and_build_dir
        "compile":{
            "learning_rate":1e-5,
            "loss":"mse",
            "metrics":["mse", "mae"],
            "weighted_metrics":["mse", "mae"],
            },

        ## Exclusive to train
        "train":{
            "early_stop_metric":"val_mse", ## metric evaluated for stagnation
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
            "train_procs":4,

            "val_files":None,
            "val_procs":2,

            "frequency":3,
            "block_size":64,
            "buf_size_mb":512,
            "deterministic":False,
            },

        "name":"test-1",
        "seed":200007221750,
        "notes":"",
        }

if __name__=="__main__":
    sequences_dir = Path("/rstor/mdodson/thesis/sequences")
    model_parent_dir = Path("data/models/new")

    config["data"]["train_files"] = [
            sequences_dir.joinpath(f"sequences_loam_se.h5")]
    config["data"]["val_files"] = [
            sequences_dir.joinpath(f"sequences_loam_se.h5")]

    """
    Declare training and validation dataset generators using the config
    """
    data_t = gen_sequence_samples(
            sequence_hdf5s=config["data"]["train_files"],
            num_procs=config["data"]["train_procs"],
            sample_on_frequency=True,
            seed=config["seed"],
            **config["feats"],
            **config["data"],
            )

    data_v = gen_sequence_samples(
            sequence_hdf5s=config["data"]["train_files"],
            num_procs=config["data"]["val_procs"],
            sample_on_frequency=True,
            seed=config["seed"],
            **config["feats"],
            **config["data"],
            )

    for (tw,th,ts,tsi,tt),tp in data_t.batch(8):
        print(tw.shape,th.shape,ts.shape,tsi.shape,tt.shape,tp.shape)
        break

    for (vw,vh,vs,vsi,vt),vp in data_v.batch(8):
        print(vw.shape,vh.shape,vs.shape,vsi.shape,vt.shape,vp.shape)
        break

    print(tsi)
    print(vsi)

    print(np.all(tw==vw))
    print(np.all(th==vh))
    print(np.all(ts==vs))
    print(np.all(tsi==vsi))
    print(np.all(tt==vt))
    print(np.all(tp==vp))

    exit(0)

    """ Initialize the model, and build its directory """
    model,md = tt.ModelDir.build_from_config(
            config,
            model_parent_dir=model_parent_dir,
            print_summary=False,
            )
