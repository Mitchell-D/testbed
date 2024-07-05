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

from generators import gen_timegrid_samples,make_sequence_hdf5
from list_feats import nldas_record_mapping,noahlsm_record_mapping
from list_feats import umd_veg_classes, statsgo_textures

if __name__=="__main__":
    gridstat_dir = Path("data/grid_stats")
    timegrid_dir = Path("/rstor/mdodson/thesis/timegrids_new")
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

    #region_substr,px_idx = "y000-098_x000-154", (49,77) ## NW
    #region_substr,px_idx = "y000-098_x154-308", (49,231) ## NC
    #region_substr,px_idx = "y000-098_x308-462", (49,385) ## NE
    #region_substr,px_idx = "y098-195_x000-154", (147,77) ## SW
    #region_substr,px_idx = "y098-195_x154-308", (147,231) ## SC
    region_substr,(yidx,xidx) = "y098-195_x308-462", (130,312) ## SE

    """ Define some conditions constraining valid samples """
    f_select_ints = "lambda a:np.any(np.stack(" + \
            "[a==v for v in {class_ints}], axis=-1), axis=-1)"
    static_conditions = [
            #("int_veg", f_select_ints.format(class_ints=(7,8,9,10))),
            ("int_soil", f_select_ints.format(class_ints=(6,))),
            #("pct_silt", "lambda a:a>=.2"),
            ("m_valid", "lambda a:a==1."),
            #("vidx", f"lambda a:a=={yidx}"),
            #("hidx", f"lambda a:a=={xidx}"),
            ]

    timegrid_paths = [
            p for p in timegrid_dir.iterdir()
            if region_substr in p.stem
            ]

    tmp_file = h5py.File(timegrid_paths[0], "r")
    region_sdata = tmp_file["/data/static"][...]
    region_slabels = json.loads(tmp_file["data"].attrs["static"])["flabels"]
    print(region_sdata[...,region_slabels.index("m_conus")])
    print(region_sdata[...,region_slabels.index("vidx")])
    print(region_sdata[...,region_slabels.index("hidx")])
    tmp_file.close()

    #seq_file_path = sequences_dir.joinpath(f"sequences_onepx_{yidx}-{xidx}.h5")
    seq_file_path = sequences_dir.joinpath(f"sequences_loam_se.h5")
    make_sequence_hdf5(
            seq_h5_path=seq_file_path,
            timegrid_paths=timegrid_paths,
            window_size=24,
            horizon_size=24*14,
            window_feats=window_feats,
            horizon_feats=horizon_feats,
            pred_feats=pred_feats,
            static_feats=static_feats,
            static_int_feats=[("int_veg",14)],
            static_conditions=static_conditions,
            num_procs=6,
            deterministic=False,
            block_size=16,
            buf_size_mb=4096,
            samples_per_timegrid=1024,
            max_offset=23,
            sample_separation=53,
            include_init_state_in_predictors=True,
            load_full_grid=False,
            seed=200007221750,

            prefetch_count=3,
            batch_size=64,
            max_batches=None,
            samples_per_chunk=256,
            debug=True,
            )
