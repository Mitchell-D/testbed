import numpy as np
import pickle as pkl
import random as rand
import json
import h5py
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
from pprint import pprint as ppt
import matplotlib.pyplot as plt
import tensorflow as tf

import model_methods as mm
import tracktrain as tt
from list_feats import dynamic_coeffs,static_coeffs
import generators

if __name__=="__main__":
    timegrid_dir = Path("data/timegrids/")
    model_parent_dir = Path("data/models/new")

    eval_regions = (
            "y000-098_x000-154",
            "y000-098_x154-308",
            "y000-098_x308-462",
            "y098-195_x000-154",
            "y098-195_x154-308",
            "y098-195_x308-462",
            )
    eval_time_substrings = tuple(map(str,range(2013,2022)))

    #start_datetime = datetime(2018,5,1)
    #end_datetime = datetime(2018,11,1)
    start_datetime = datetime(2018,1,1)
    end_datetime = datetime(2021,12,16)

    model_name = "lstm-16"
    weights_file = "lstm-16_505_0.047.weights.h5"

    batch_size=2048
    buf_size_mb=128
    num_procs = 7


    """
    Get lists of timegrids per region, relying on the expected naming
    scheme timegrid_{YYYY}q{Q}_y{vmin}-{vmax}_x{hmin}-{hmax}.h5
    """
    timegrid_paths = {
            rs:sorted([
                (tg,tuple(tg_tup)) for tg,tg_tup in map(
                    lambda p:(p,p.stem.split("_")),
                    timegrid_dir.iterdir())
                if tg_tup[0] == "timegrid"
                and any(ss in tg_tup[1] for ss in eval_time_substrings)
                and tg_tup[2] in rs
                and tg_tup[3] in rs
                ])
            for rs in eval_regions
            }

    """ Load a specific trained model's ModelDir for evaluation """
    md = tt.ModelDir(
            model_parent_dir.joinpath(model_name),
            custom_model_builders={
                "lstm-s2s":lambda args:mm.get_lstm_s2s(**args),
                })
    ppt(md.config)

    for tmp_region,v in timegrid_paths.items():
        rpaths,rtups = zip(*v)
        #'''
        gen_tg = generators.gen_timegrid_subgrids(
                timegrid_paths=rpaths,
                window_size=md.config["model"]["window_size"],
                horizon_size=md.config["model"]["horizon_size"],
                window_feats=md.config["feats"]["window_feats"],
                horizon_feats=md.config["feats"]["horizon_feats"],
                pred_feats=md.config["feats"]["pred_feats"],
                static_feats=md.config["feats"]["static_feats"] + ["m_valid"],
                static_int_feats=[("int_veg",14)],
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                frequency=7*24,
                vidx_min=10,
                vidx_max=58,
                hidx_min=10,
                hidx_max=58,
                buf_size_mb=512,
                load_full_grid=False,
                include_init_state_in_predictors=True,
                seed=200007221750,
                )
        #'''
        max_count = 5
        counter = 0
        m_valid = None
        for (w,h,s,si,t),p in gen_tg:
            if m_valid is None:
                m_valid = s[...,-1].astype(bool)
            s = s[...,:-1]
            print(w.shape, h.shape, s.shape, si.shape, t.shape, p.shape)
            print(w[:,m_valid].shape, h[:,m_valid].shape, p[:,m_valid].shape)
            counter += 1
            if counter == max_count:
                break
        break
