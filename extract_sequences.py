"""
Methods for interacting with 'timegrid' style HDF5s, which each cover 1/6 of
CONUS over a 3 month period, and store their data as a (T,P,Q,F) dynamic grid
with (P,Q,F) static grids and (T,1) timestamps
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
import tensorflow as tf
from pprint import pprint as ppt

from generators import make_sequence_hdf5,timegrid_sequence_dataset
from list_feats import nldas_record_mapping,noahlsm_record_mapping
from list_feats import umd_veg_classes, statsgo_textures,derived_feats
from eval_timegrid import parse_timegrid_path

if __name__=="__main__":
    gridstat_dir = Path("data/grid_stats")
    timegrid_dir = Path("/rstor/mdodson/thesis/timegrids")
    sequences_dir = Path("/rstor/mdodson/thesis/sequences")

    window_feats = [
            "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
            "dlwrf", "dswrf", "apcp",
            "soilm-10", "soilm-40", "soilm-100", "soilm-200", "weasd",
            ]
    horizon_feats = [
            "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
            "dlwrf", "dswrf", "apcp",
            ]
    pred_feats = [ 'soilm-10', 'soilm-40', 'soilm-100', 'soilm-200', "weasd" ]
    static_feats = [ "pct_sand", "pct_silt", "pct_clay", "elev", "elev_std" ]
    int_feats = [ "int_veg" ]

    #region_substr,region_label = "y000-098_x000-154","nw"
    #region_substr,region_label = "y000-098_x154-308","nc"
    #region_substr,region_label = "y000-098_x308-462","ne"

    #region_substr,region_label = "y098-195_x000-154","sw"
    #region_substr,region_label = "y098-195_x154-308","sc"
    region_substr,region_label = "y098-195_x308-462","se"

    ## range of valid file years to include
    year_range = (2013,2018)
    #year_range = (2018,2023)

    #valid_seasons,season_label = (1,2,3,4),"all"
    #valid_seasons,season_label = (1,4),"cold"
    valid_seasons,season_label = (2,3),"warm"

    """ Define some conditions constraining valid samples """
    f_select_ints = "lambda a:np.any(np.stack(" + \
            "[a==v for v in {class_ints}], axis=-1), axis=-1)"
    static_conditions = [
            #("int_veg", f_select_ints.format(class_ints=(7,8,9,10))),
            ("int_soil", f_select_ints.format(class_ints=(3,))),
            #("pct_silt", "lambda a:a>=.2"),
            ("m_valid", "lambda a:a==1."),
            #("vidx", f"lambda a:a=={yidx}"),
            #("hidx", f"lambda a:a=={xidx}"),
            ]

    region_label += "-sandyloam"

    timegrid_paths = [
            (*parse_timegrid_path(p),p) for p in timegrid_dir.iterdir()
            if region_substr in p.stem
            ]

    seq_path = sequences_dir.joinpath(
            f"sequences_{region_label}_{season_label}" + \
                    f"_{'-'.join(map(str,year_range))}.h5")

    #'''
    pred_feats += ["rsm-10","rsm-40","rsm-100","rsm-200","rsm-fc","soilm-fc"]
    ds = timegrid_sequence_dataset(
            timegrid_paths=[
                p for (year,quarter),_,_,p in timegrid_paths
                if quarter in valid_seasons and year in range(*year_range)
                ],
            window_size=24,
            horizon_size=24*14,
            window_feats=window_feats,
            horizon_feats=horizon_feats,
            pred_feats=pred_feats,
            static_feats=static_feats,
            static_int_feats=[("int_veg",14)],
            static_conditions=static_conditions,
            derived_feats=derived_feats,
            num_procs=1,
            deterministic=False,
            block_size=16,
            buf_size_mb=4096,
            samples_per_timegrid=2**16,
            max_offset=23,
            sample_separation=31,
            include_init_state_in_predictors=True,
            load_full_grid=False,
            seed=200007221750,
            )
    ppt(list(enumerate(pred_feats)))
    for (w,h,s,si,t),p in ds:
        for j in range(p.shape[0]):
            print([f"{v:.3E}" for v in p[j]])
        break
    #'''

    '''
    make_sequence_hdf5(
            ## args passed to timegrid_sequence_dataset
            seq_h5_path=seq_path,
            timegrid_paths=[
                p for (year,quarter),_,_,p in timegrid_paths
                if quarter in valid_seasons and year in range(*year_range)
                ],
            window_size=24,
            horizon_size=24*14,
            window_feats=window_feats,
            horizon_feats=horizon_feats,
            pred_feats=pred_feats,
            static_feats=static_feats,
            static_int_feats=[("int_veg",14)],
            static_conditions=static_conditions,
            derived_feats=derived_feats,
            num_procs=8,
            deterministic=False,
            block_size=16,
            buf_size_mb=4096,
            samples_per_timegrid=2**16,
            max_offset=23,
            sample_separation=31,
            include_init_state_in_predictors=True,
            load_full_grid=False,
            seed=200007221750,

            ## args for hdf5 builder
            prefetch_count=3,
            batch_size=64,
            max_batches=None,
            samples_per_chunk=128,
            debug=True,
            )
    '''
